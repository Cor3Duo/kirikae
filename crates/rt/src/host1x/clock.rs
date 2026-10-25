//! Presentation clock shared by the video pipeline (NVDEC → VIC → VI) and the
//! audio sink.
//!
//! NVDEC counts the pictures it produces, VIC marks the surfaces that came from
//! a YUV source, and VI asks the clock whether a frame is due before it presents
//! it. While audio is playing, the reference is the position of the samples
//! handed to the audio device; when audio is silent, muted or has stopped
//! reporting, the clock falls back to a monotonic schedule anchored at the first
//! frame of the current media segment.
//!
//! # Timestamps
//!
//! VI receives an Android `QueueBufferInput` in `QueueBuffer`; its first field
//! is the buffer timestamp (`-1` means "automatic", see
//! [`Timestamp::Automatic`]). Titles that play video fill it from the container;
//! titles that do not get a synthesized timestamp: the previous frame plus the
//! nominal cadence ([`NOMINAL_FPS`]), which is the frame-index × 1/fps source.
//!
//! # Broken clocks this handles
//!
//! * First frame of a segment: the schedule is anchored to it, so it presents
//!   immediately with no offset and zero drift.
//! * A backward jump or a forward jump larger than a second is a seek: the
//!   segment is re-anchored to the new timestamp and the offset is reset.
//! * A repeated timestamp (or no picture decoded since the last one) is a
//!   starved decoder: the frame is held until the next slot, at most
//!   [`MAX_WAIT_FACTOR`] periods, and counted as repeated.
//! * A frame more than [`DROP_LIMIT_FACTOR`] periods late is dropped instead of
//!   sleeping, and the segment is rebased so the next frame can catch up.
//! * Headless or `KIRIKAE_NO_FRAME_LIMIT` runs never wait, and a muted or stalled
//!   audio sink falls back to the monotonic clock.
//!
//! # Limits
//!
//! The audio position is the point at which samples are handed to the device, so
//! the device's own buffering latency (one device buffer) is not compensated.
//! The automatic cadence is [`NOMINAL_FPS`] because the NVDEC picture setup does
//! not carry a frame duration; a title that wants another rate has to stamp its
//! buffers, or the caller can change the cadence with
//! [`PresentationClock::set_rate`]. Video the guest presents through NVN instead
//! of VI never reaches this clock.
//!
//! Counters are printed with `KIRIKAE_STATS=1`; they include the YUV layers VI
//! queued, presented, dropped, converted to RGB and failed to convert.

use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// Cadence assumed when the guest does not provide a container timestamp.
pub const NOMINAL_FPS: f64 = 30.0;
/// A forward or backward jump larger than this starts a new media segment.
const SEEK_JUMP_NS: i64 = 1_000_000_000;
/// A repeat is held for at most this many frame periods, so a starved decoder
/// never blocks the display thread indefinitely.
const MAX_WAIT_FACTOR: i64 = 2;
/// A frame later than this many periods is dropped instead of presented late.
const DROP_LIMIT_FACTOR: i64 = 3;
/// The audio position is considered live for this long after the last report.
const AUDIO_STALE: Duration = Duration::from_millis(500);
/// A second source is ignored while the current one has reported this recently.
const AUDIO_SOURCE_TIMEOUT: Duration = Duration::from_millis(250);
/// How long a VIC output surface stays tagged as video.
const SURFACE_TTL: Duration = Duration::from_secs(2);
/// A composed 4:2:0 surface tags its luma and both chroma planes, so a few
/// compositions in flight need more slots than one tag each.
const SURFACES_MAX: usize = 64;
/// Video frames between `KIRIKAE_STATS` lines (about five seconds at 30 fps).
const STATS_INTERVAL: u64 = 150;

/// Which audio service owns the playback position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioSource {
    /// `audout:u/IAudioOut`, fed by the sink thread.
    AudOut,
    /// `audren:u/IAudioRenderer`, fed by the mixer render loop.
    AudRen,
}

/// Timestamp attached to a VI buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Timestamp {
    /// Container timestamp in nanoseconds, as provided by the guest.
    Guest(i64),
    /// The guest asked for an automatic timestamp (or provided none), so the
    /// clock derives it from the frame index and the nominal cadence.
    Automatic,
}

/// What the caller should do with the frame it asked about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Present now.
    Present,
    /// Sleep for this long, then present.
    Wait(Duration),
    /// Skip this frame; the next one will be scheduled normally.
    Drop,
}

/// Decision returned by [`PresentationClock::plan`].
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub action: Action,
    /// Media timestamp this plan was computed for.
    pub pts: i64,
    /// Host instant the frame should reach the display.
    pub deadline: Instant,
    /// `now - deadline` in nanoseconds (positive means late).
    pub drift_ns: i64,
    /// The frame carries no new content (starved decoder or repeated buffer).
    pub repeat: bool,
}

/// Counters exposed by `KIRIKAE_STATS`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub decoded: u64,
    pub composed: u64,
    pub presented: u64,
    pub early: u64,
    pub late: u64,
    pub dropped: u64,
    pub repeated: u64,
    pub discontinuities: u64,
    /// YUV layers queued by VI.
    pub yuv_queued: u64,
    /// YUV layers that reached the display path.
    pub yuv_presented: u64,
    /// YUV layers VI dropped because the presentation clock was too far behind.
    pub yuv_dropped: u64,
    /// YUV layers decoded and converted to RGB for the swapchain.
    pub converted: u64,
    /// YUV layers queued but not presented (unsupported layout, unmapped buffer).
    pub failed: u64,
    /// Drift of the last presented (or dropped) frame, in nanoseconds.
    pub drift_ns: i64,
    /// Largest absolute drift seen, in nanoseconds.
    pub max_drift_ns: i64,
}

#[derive(Clone, Copy, Debug)]
struct Segment {
    /// First timestamp of this segment.
    pts: i64,
    /// Host instant at which `pts` is due (monotonic schedule).
    host: Instant,
    /// `audio_media - pts` measured when the segment was anchored. Present only
    /// when the segment is driven by the audio clock.
    skew: Option<i64>,
}

#[derive(Clone, Copy, Debug)]
struct Pending {
    pts: i64,
    deadline: Instant,
    repeat: bool,
}

#[derive(Clone, Copy, Debug)]
struct VideoSurface {
    base: u64,
    len: u64,
    at: Instant,
}

#[derive(Clone, Copy, Debug)]
struct AudioTimeline {
    active: bool,
    source: Option<AudioSource>,
    rate: u32,
    anchor_media_ns: i64,
    anchor_host: Instant,
    last_seen: Instant,
}

impl AudioTimeline {
    fn new(now: Instant) -> AudioTimeline {
        AudioTimeline {
            active: false,
            source: None,
            rate: 0,
            anchor_media_ns: 0,
            anchor_host: now,
            last_seen: now,
        }
    }

    /// Reports the media position of the audio sink. `playing` is false when the
    /// sink is stopped, starved or muted; the clock then ignores the audio and
    /// falls back to the monotonic schedule.
    fn observe(
        &mut self,
        played: u64,
        rate: u32,
        playing: bool,
        source: AudioSource,
        now: Instant,
    ) {
        if !playing || rate == 0 {
            if self.source.is_none() || self.source == Some(source) {
                self.active = false;
            }
            return;
        }
        if let Some(current) = self.source
            && current != source
            && self.active
            && now.saturating_duration_since(self.last_seen) < AUDIO_SOURCE_TIMEOUT
        {
            // The other service is still reporting; ignore this one.
            return;
        }
        self.active = true;
        self.source = Some(source);
        self.rate = rate;
        self.anchor_media_ns = media_ns(played, rate);
        self.anchor_host = now;
        self.last_seen = now;
    }

    fn media_now(&self, now: Instant) -> Option<i64> {
        if !self.active || now.saturating_duration_since(self.last_seen) > AUDIO_STALE {
            return None;
        }
        Some(
            self.anchor_media_ns
                .saturating_add(since(self.anchor_host, now)),
        )
    }
}

/// Media clock for decoded video.
pub struct PresentationClock {
    fps: f64,
    period_ns: i64,
    blocking: bool,
    audio: AudioTimeline,
    audio_was_active: bool,
    segment: Option<Segment>,
    last_pts: Option<i64>,
    last_deadline: Option<Instant>,
    decoded_at_present: u64,
    pending: Option<Pending>,
    surfaces: Vec<VideoSurface>,
    counters: Counters,
}

impl Default for PresentationClock {
    fn default() -> Self {
        Self::new()
    }
}

impl PresentationClock {
    pub fn new() -> PresentationClock {
        let now = Instant::now();
        PresentationClock {
            fps: NOMINAL_FPS,
            period_ns: (1e9 / NOMINAL_FPS).round() as i64,
            blocking: true,
            audio: AudioTimeline::new(now),
            audio_was_active: false,
            segment: None,
            last_pts: None,
            last_deadline: None,
            decoded_at_present: 0,
            pending: None,
            surfaces: Vec::new(),
            counters: Counters::default(),
        }
    }

    /// Disables waiting. Headless runs and `KIRIKAE_NO_FRAME_LIMIT` set this.
    pub fn set_blocking(&mut self, blocking: bool) {
        self.blocking = blocking;
    }

    /// Changes the nominal cadence used for synthesized timestamps and for the
    /// repeat/late thresholds. A guest-provided timestamp already carries its
    /// own timing; the segment is rebased so the change does not jump.
    pub fn set_rate(&mut self, fps: f64, now: Instant) {
        if !fps.is_finite() || fps <= 0.0 {
            return;
        }
        self.fps = fps.clamp(1.0, 240.0);
        self.period_ns = (1e9 / self.fps).round() as i64;
        if let Some(last) = self.last_pts {
            self.segment = Some(Segment {
                pts: last,
                host: self.last_deadline.unwrap_or(now),
                skew: self.segment.and_then(|s| s.skew),
            });
        }
    }

    /// Counts a decoded NVDEC picture.
    pub fn decoded(&mut self) {
        self.counters.decoded += 1;
    }

    /// Counts a YUV layer queued by VI.
    pub fn yuv_queued(&mut self) {
        self.counters.yuv_queued += 1;
    }

    /// Counts a YUV layer that reached the display path.
    pub fn yuv_presented(&mut self) {
        self.counters.yuv_presented += 1;
    }

    /// Counts a YUV layer the clock dropped as too late.
    pub fn yuv_dropped(&mut self) {
        self.counters.yuv_dropped += 1;
    }

    /// Counts a YUV layer that was converted to RGB for the swapchain.
    pub fn converted(&mut self) {
        self.counters.converted += 1;
    }

    /// Counts a YUV layer that could not be presented.
    pub fn failed(&mut self) {
        self.counters.failed += 1;
    }

    /// Tags a VIC output surface as carrying video, so VI can gate an RGBA
    /// presentation that was composed from a YUV source.
    pub fn composed_surface(&mut self, base: u64, len: usize, now: Instant) {
        self.counters.composed += 1;
        self.tag_video_surface(base, len, now);
    }

    /// Tags one plane of a composed surface without counting another
    /// composition. Used for the chroma planes, which may live in a separate
    /// mapping from the luma plane.
    pub fn tag_video_surface(&mut self, base: u64, len: usize, now: Instant) {
        self.expire_surfaces(now);
        if self.surfaces.len() == SURFACES_MAX {
            self.surfaces.remove(0);
        }
        self.surfaces.push(VideoSurface {
            base,
            len: len as u64,
            at: now,
        });
    }

    pub fn is_video_surface(&mut self, addr: u64, now: Instant) -> bool {
        self.expire_surfaces(now);
        self.surfaces
            .iter()
            .any(|s| addr >= s.base && addr < s.base.saturating_add(s.len))
    }

    fn expire_surfaces(&mut self, now: Instant) {
        self.surfaces
            .retain(|s| now.saturating_duration_since(s.at) < SURFACE_TTL);
    }

    /// Reports the audio device position. Called from the sink/render thread.
    pub fn audio_position(
        &mut self,
        played: u64,
        rate: u32,
        playing: bool,
        source: AudioSource,
        now: Instant,
    ) {
        self.audio.observe(played, rate, playing, source, now);
    }

    /// Computes when a frame with `ts` should be presented.
    pub fn plan(&mut self, ts: Timestamp, now: Instant) -> Plan {
        let pts = match ts {
            Timestamp::Guest(pts) => pts,
            Timestamp::Automatic => self
                .last_pts
                .map_or(0, |last| last.saturating_add(self.period_ns)),
        };
        let mut repeat = false;
        if let Some(last) = self.last_pts {
            let delta = pts.saturating_sub(last);
            if !(-SEEK_JUMP_NS..=SEEK_JUMP_NS).contains(&delta) {
                // Seek or discontinuity: start a new segment at this timestamp.
                self.counters.discontinuities += 1;
                self.segment = None;
            } else if delta <= 0 {
                // Same frame again, or a timestamp out of order: hold the slot.
                repeat = true;
            }
        }
        if matches!(ts, Timestamp::Automatic) && self.counters.decoded == self.decoded_at_present {
            // Nothing new came out of NVDEC: the decoder is starved.
            repeat = true;
        }
        let audio_now = self.audio.media_now(now);
        if self.segment.is_none() {
            self.segment = Some(Segment {
                pts,
                host: now,
                skew: audio_now.map(|audio| audio.saturating_sub(pts)),
            });
        }
        let audio_active = audio_now.is_some();
        if audio_active != self.audio_was_active {
            self.audio_was_active = audio_active;
            self.segment = match audio_now {
                Some(audio) => Some(Segment {
                    pts,
                    host: now,
                    skew: Some(audio.saturating_sub(pts)),
                }),
                None => Some(Segment {
                    pts: self.last_pts.unwrap_or(pts),
                    host: self.last_deadline.unwrap_or(now),
                    skew: None,
                }),
            };
        } else if audio_active
            && let Some(segment) = &mut self.segment
            && segment.skew.is_none()
        {
            segment.skew = audio_now.map(|audio| audio.saturating_sub(pts));
        }
        let segment = self.segment.unwrap_or(Segment {
            pts,
            host: now,
            skew: None,
        });
        let period = Duration::from_nanos(self.period_ns.unsigned_abs());
        let deadline = if repeat {
            self.last_deadline
                .unwrap_or(now)
                .checked_add(period)
                .unwrap_or(now)
        } else if let (Some(skew), Some(audio)) = (segment.skew, audio_now) {
            shift(now, pts.saturating_add(skew).saturating_sub(audio))
        } else {
            shift(segment.host, pts.saturating_sub(segment.pts))
        };
        let drift_ns = since(now, deadline);
        let max_wait = Duration::from_nanos((self.period_ns * MAX_WAIT_FACTOR).unsigned_abs());
        let drop_limit = self.period_ns * DROP_LIMIT_FACTOR;
        let action = if !self.blocking {
            Action::Present
        } else if !repeat && drift_ns > drop_limit {
            Action::Drop
        } else if now < deadline {
            let wait = deadline - now;
            Action::Wait(wait.min(max_wait))
        } else {
            Action::Present
        };
        self.pending = Some(Pending {
            pts,
            deadline,
            repeat,
        });
        Plan {
            action,
            pts,
            deadline,
            drift_ns,
            repeat,
        }
    }

    /// Records that the frame from the last [`Self::plan`] reached the display
    /// at `now`, and returns its drift.
    pub fn presented(&mut self, now: Instant) -> i64 {
        let Some(pending) = self.pending.take() else {
            return 0;
        };
        let drift = since(now, pending.deadline);
        self.counters.presented += 1;
        if pending.repeat {
            self.counters.repeated += 1;
        }
        let half = self.period_ns / 2;
        if drift > half {
            self.counters.late += 1;
        } else if drift < -half {
            self.counters.early += 1;
        }
        self.note_drift(drift);
        self.last_pts = Some(pending.pts);
        self.last_deadline = Some(pending.deadline);
        self.decoded_at_present = self.counters.decoded;
        drift
    }

    /// Records that the frame from the last [`Self::plan`] was dropped because
    /// it was too late. The segment is rebased so the next frame catches up
    /// instead of being dropped forever.
    pub fn dropped(&mut self, now: Instant) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        self.counters.dropped += 1;
        let drift = since(now, pending.deadline);
        self.note_drift(drift);
        if let Some(segment) = &mut self.segment {
            if let Some(skew) = segment.skew {
                segment.skew = Some(skew.saturating_add(drift));
            }
            segment.pts = pending.pts;
            segment.host = now;
        }
        self.last_deadline = Some(now);
    }

    fn note_drift(&mut self, drift: i64) {
        self.counters.drift_ns = drift;
        self.counters.max_drift_ns = self.counters.max_drift_ns.max(drift.saturating_abs());
    }

    pub fn counters(&self) -> Counters {
        self.counters
    }

    /// `"audio"` while the sink drives the clock, `"monotonic"` otherwise.
    pub fn source(&self) -> &'static str {
        if self.audio.media_now(self.audio.last_seen).is_some() {
            "audio"
        } else {
            "monotonic"
        }
    }
}

fn media_ns(frames: u64, rate: u32) -> i64 {
    if rate == 0 {
        return 0;
    }
    let ns = u128::from(frames) * 1_000_000_000 / u128::from(rate);
    ns.min(i64::MAX as u128) as i64
}

/// `a - b` in nanoseconds, saturating.
fn since(a: Instant, b: Instant) -> i64 {
    let ns = |d: Duration| d.as_nanos().min(i64::MAX as u128) as i64;
    if a >= b { ns(a - b) } else { -ns(b - a) }
}

fn shift(base: Instant, ns: i64) -> Instant {
    let delta = Duration::from_nanos(ns.unsigned_abs());
    let shifted = if ns >= 0 {
        base.checked_add(delta)
    } else {
        base.checked_sub(delta)
    };
    shifted.unwrap_or(base)
}

// ---------------------------------------------------------------- process-wide clock

static CLOCK: OnceLock<Mutex<PresentationClock>> = OnceLock::new();

fn global() -> &'static Mutex<PresentationClock> {
    CLOCK.get_or_init(|| {
        let mut clock = PresentationClock::new();
        clock.set_blocking(
            !crate::display::headless() && std::env::var_os("KIRIKAE_NO_FRAME_LIMIT").is_none(),
        );
        Mutex::new(clock)
    })
}

fn lock() -> MutexGuard<'static, PresentationClock> {
    global().lock().unwrap_or_else(|e| e.into_inner())
}

fn stats_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KIRIKAE_STATS").is_some())
}

fn muted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("KIRIKAE_MUTE").is_some_and(|v| v != "0"))
}

/// Reports the device position of the audio sink.
pub fn audio_position(played: u64, rate: u32, playing: bool, source: AudioSource) {
    let playing = playing && !muted();
    lock().audio_position(played, rate, playing, source, Instant::now());
}

/// Counts a decoded NVDEC picture.
pub fn decoded() {
    lock().decoded();
}

/// Counts a YUV layer queued by VI.
pub fn yuv_queued() {
    lock().yuv_queued();
}

/// Counts a YUV layer that reached the display path.
pub fn yuv_presented() {
    lock().yuv_presented();
}

/// Counts a YUV layer dropped as too late.
pub fn yuv_dropped() {
    lock().yuv_dropped();
}

/// Counts a YUV layer converted to RGB for the swapchain.
pub fn converted() {
    lock().converted();
}

/// Counts a YUV layer that could not be presented.
pub fn yuv_failed() {
    lock().failed();
}

/// Tags a VIC output surface as video.
pub fn composed_surface(base: u64, len: usize) {
    lock().composed_surface(base, len, Instant::now());
}

/// Tags one more plane of a composed surface.
pub fn tag_video_surface(base: u64, len: usize) {
    lock().tag_video_surface(base, len, Instant::now());
}

/// Whether `addr` is a surface VIC recently composed from a YUV source.
pub fn is_video_surface(addr: u64) -> bool {
    lock().is_video_surface(addr, Instant::now())
}

/// Gates one video frame. Returns `true` when the caller should present it and
/// `false` when the frame was dropped as too late.
pub fn gate(ts: Timestamp) -> bool {
    let now = Instant::now();
    let mut clock = lock();
    let plan = clock.plan(ts, now);
    match plan.action {
        Action::Drop => {
            clock.dropped(now);
            report(&clock);
            false
        }
        Action::Present => {
            clock.presented(now);
            report(&clock);
            true
        }
        Action::Wait(wait) => {
            drop(clock);
            if !wait.is_zero() {
                std::thread::sleep(wait);
            }
            let now = Instant::now();
            let mut clock = lock();
            clock.presented(now);
            report(&clock);
            true
        }
    }
}

fn report(clock: &PresentationClock) {
    if !stats_enabled() {
        return;
    }
    let c = clock.counters();
    let due = |n: u64| n != 0 && (n == 1 || n.is_multiple_of(STATS_INTERVAL));
    let yuv = c.yuv_queued + c.yuv_presented + c.yuv_dropped + c.converted + c.failed;
    if !due(c.presented + c.dropped) && !due(yuv) {
        return;
    }
    eprintln!(
        "kirikae: av: {} decoded, {} composed, {} presented ({} early, {} late, {} dropped, {} repeated, {} seeks), drift {:.1} ms (max {:.1} ms), {} clock; YUV {} queued, {} presented, {} dropped, {} converted, {} failed",
        c.decoded,
        c.composed,
        c.presented,
        c.early,
        c.late,
        c.dropped,
        c.repeated,
        c.discontinuities,
        c.drift_ns as f64 / 1e6,
        c.max_drift_ns as f64 / 1e6,
        clock.source(),
        c.yuv_queued,
        c.yuv_presented,
        c.yuv_dropped,
        c.converted,
        c.failed
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const NS_PER_FRAME: i64 = 1_000_000_000 / 30;
    const SAMPLES_PER_FRAME: u64 = 48_000 / 30;

    fn base() -> Instant {
        Instant::now()
    }

    fn at(base: Instant, ns: i64) -> Instant {
        shift(base, ns)
    }

    fn present(clock: &mut PresentationClock, pts: i64, now: Instant) {
        let plan = clock.plan(Timestamp::Guest(pts), now);
        assert_ne!(plan.action, Action::Drop, "unexpected drop at {pts}");
        clock.presented(now);
    }

    #[test]
    fn thirty_fps_presents_on_schedule_against_48khz_audio() {
        let base = base();
        let mut clock = PresentationClock::new();
        clock.audio_position(0, 48_000, true, AudioSource::AudRen, base);
        for i in 0..30i64 {
            let pts = i * NS_PER_FRAME;
            let now = at(base, pts);
            clock.audio_position(
                i as u64 * SAMPLES_PER_FRAME,
                48_000,
                true,
                AudioSource::AudRen,
                now,
            );
            let plan = clock.plan(Timestamp::Guest(pts), now);
            match plan.action {
                Action::Present => {}
                Action::Wait(wait) => assert!(wait < Duration::from_millis(1), "{wait:?}"),
                Action::Drop => panic!("frame {i} dropped"),
            }
            clock.presented(now);
        }
        let c = clock.counters();
        assert_eq!(c.presented, 30);
        assert_eq!(c.dropped, 0);
        assert_eq!(c.repeated, 0);
        assert_eq!(c.discontinuities, 0);
        assert!(c.drift_ns.abs() < 1_000_000, "drift {} ns", c.drift_ns);
        assert_eq!(clock.source(), "audio");
    }

    #[test]
    fn a_late_frame_is_dropped_instead_of_stalling() {
        let base = base();
        let mut clock = PresentationClock::new();
        clock.audio_position(0, 48_000, true, AudioSource::AudRen, base);
        present(&mut clock, 0, base);
        // The decoder is 200 ms behind the audio clock.
        let now = at(base, 200_000_000);
        clock.audio_position(200 * 48, 48_000, true, AudioSource::AudRen, now);
        let plan = clock.plan(Timestamp::Guest(NS_PER_FRAME), now);
        assert_eq!(plan.action, Action::Drop);
        assert!(plan.drift_ns > 100_000_000, "drift {} ns", plan.drift_ns);
        clock.dropped(now);
        assert_eq!(clock.counters().dropped, 1);
        // The drop rebases the segment, so the next frame waits only briefly.
        let next = at(now, NS_PER_FRAME);
        clock.audio_position(233 * 48, 48_000, true, AudioSource::AudRen, next);
        let plan = clock.plan(Timestamp::Guest(2 * NS_PER_FRAME), next);
        assert_ne!(plan.action, Action::Drop);
        if let Action::Wait(wait) = plan.action {
            assert!(wait <= Duration::from_millis(1), "{wait:?}");
        }
    }

    #[test]
    fn a_starved_decoder_repeats_without_deadlocking() {
        let base = base();
        let mut clock = PresentationClock::new();
        clock.audio_position(0, 48_000, true, AudioSource::AudRen, base);
        present(&mut clock, 0, base);
        // Same timestamp: no picture came out of the decoder.
        let now = at(base, 10_000_000);
        clock.audio_position(480, 48_000, true, AudioSource::AudRen, now);
        let plan = clock.plan(Timestamp::Guest(0), now);
        let Action::Wait(wait) = plan.action else {
            panic!("starved frame should be held, got {:?}", plan.action);
        };
        let cap = Duration::from_nanos((NS_PER_FRAME * MAX_WAIT_FACTOR) as u64);
        assert!(wait <= cap, "{wait:?} exceeds {cap:?}");
        clock.presented(at(now, wait.as_nanos() as i64));
        assert_eq!(clock.counters().repeated, 1);
        // A starved decoder that falls far behind the audio still converges.
        let now = at(base, 1_000_000_000);
        clock.audio_position(48_000, 48_000, true, AudioSource::AudRen, now);
        let plan = clock.plan(Timestamp::Guest(0), now);
        assert_ne!(plan.action, Action::Drop);
        if let Action::Wait(wait) = plan.action {
            assert!(wait <= cap);
        }
        clock.presented(now);
        assert!(clock.counters().repeated >= 2);
    }

    #[test]
    fn muted_or_headless_plans_never_stall() {
        let base = base();
        let mut clock = PresentationClock::new();
        clock.audio_position(0, 48_000, false, AudioSource::AudRen, base);
        assert_eq!(clock.source(), "monotonic");
        // Headless: a frame due in the future is presented immediately.
        clock.set_blocking(false);
        let first = clock.plan(Timestamp::Guest(0), base);
        assert_eq!(first.action, Action::Present);
        clock.presented(base);
        let early = clock.plan(Timestamp::Guest(500_000_000), at(base, 100_000_000));
        assert_eq!(early.action, Action::Present);
        assert!(
            early.drift_ns <= -400_000_000,
            "drift {} ns",
            early.drift_ns
        );
        clock.presented(at(base, 100_000_000));
        assert_eq!(clock.counters().early, 1);
        // With the clock blocked, a stalled audio sink still falls back to a
        // bounded monotonic wait instead of an unbounded one.
        let mut clock = PresentationClock::new();
        clock.audio_position(0, 48_000, false, AudioSource::AudRen, base);
        present(&mut clock, 0, base);
        let plan = clock.plan(Timestamp::Guest(NS_PER_FRAME), at(base, 20_000_000));
        if let Action::Wait(wait) = plan.action {
            assert!(wait <= Duration::from_nanos((NS_PER_FRAME * MAX_WAIT_FACTOR) as u64));
        }
        assert_eq!(clock.source(), "monotonic");
    }

    #[test]
    fn a_seek_resets_the_offset() {
        let base = base();
        let mut clock = PresentationClock::new();
        clock.audio_position(0, 48_000, true, AudioSource::AudRen, base);
        present(&mut clock, 0, base);
        let now = at(base, NS_PER_FRAME);
        clock.audio_position(SAMPLES_PER_FRAME, 48_000, true, AudioSource::AudRen, now);
        present(&mut clock, NS_PER_FRAME, now);
        // Forward seek to ten seconds.
        let now = at(base, 10_000_000_000);
        clock.audio_position(10 * 48_000, 48_000, true, AudioSource::AudRen, now);
        let plan = clock.plan(Timestamp::Guest(10_000_000_000), now);
        assert_eq!(plan.action, Action::Present);
        assert_eq!(clock.counters().discontinuities, 1);
        clock.presented(now);
        // Backward seek to five seconds.
        let now = at(now, NS_PER_FRAME);
        clock.audio_position(5 * 48_000, 48_000, true, AudioSource::AudRen, now);
        let plan = clock.plan(Timestamp::Guest(5_000_000_000), now);
        assert_eq!(plan.action, Action::Present);
        assert_eq!(clock.counters().discontinuities, 2);
    }

    #[test]
    fn the_first_frame_presents_without_an_offset() {
        let base = base();
        let mut clock = PresentationClock::new();
        let pts = 123_456_789;
        let plan = clock.plan(Timestamp::Guest(pts), base);
        assert_eq!(plan.action, Action::Present);
        assert_eq!(plan.pts, pts);
        assert_eq!(plan.drift_ns, 0);
        clock.presented(base);
        assert_eq!(clock.counters().presented, 1);
    }

    #[test]
    fn a_rate_change_rebases_the_automatic_cadence() {
        let base = base();
        let mut clock = PresentationClock::new();
        clock.set_blocking(false);
        let first = clock.plan(Timestamp::Automatic, base);
        assert_eq!(first.pts, 0);
        clock.presented(base);
        let second = clock.plan(Timestamp::Automatic, at(base, NS_PER_FRAME));
        assert_eq!(second.pts, NS_PER_FRAME);
        clock.presented(at(base, NS_PER_FRAME));
        clock.set_rate(60.0, at(base, NS_PER_FRAME));
        let third = clock.plan(Timestamp::Automatic, at(base, NS_PER_FRAME + 16_666_667));
        assert_eq!(third.pts, NS_PER_FRAME + 16_666_667);
    }

    #[test]
    fn composed_yuv_surfaces_expire() {
        let base = base();
        let mut clock = PresentationClock::new();
        clock.composed_surface(0x8000_0000, 0x1000, base);
        assert!(clock.is_video_surface(0x8000_0000, base));
        assert!(clock.is_video_surface(0x8000_0fff, base));
        assert!(!clock.is_video_surface(0x8000_1000, base));
        assert!(!clock.is_video_surface(0x8000_0000, at(base, 3_000_000_000)));
        assert_eq!(clock.counters().composed, 1);
    }

    #[test]
    fn every_plane_of_a_composed_surface_is_tagged() {
        let base = base();
        let mut clock = PresentationClock::new();
        clock.composed_surface(0x8000_0000, 0x100, base);
        clock.tag_video_surface(0x9000_0000, 0x80, base);
        assert!(clock.is_video_surface(0x8000_0000, base));
        assert!(clock.is_video_surface(0x9000_0040, base));
        // Tagging the chroma plane does not count a second composition.
        assert_eq!(clock.counters().composed, 1);
    }

    #[test]
    fn yuv_counters_track_the_video_path() {
        let mut clock = PresentationClock::new();
        clock.yuv_queued();
        clock.yuv_presented();
        clock.yuv_dropped();
        clock.converted();
        clock.failed();
        let c = clock.counters();
        assert_eq!(c.yuv_queued, 1);
        assert_eq!(c.yuv_presented, 1);
        assert_eq!(c.yuv_dropped, 1);
        assert_eq!(c.converted, 1);
        assert_eq!(c.failed, 1);
    }
}
