//! Opus decoding for the `hwopus` service.
//!
//! The service packet format is the one described by Ryujinx's
//! `HardwareOpusDecoder`: an 8-byte header with the payload length and the final
//! range, both big endian, followed by one Opus packet:
//!
//! ```text
//! 0x00 u32be payload length
//! 0x04 u32be final range
//! 0x08 ...    Opus packet
//! ```
//!
//! Decoding uses the reference libopus through the `opus` crate. Mono and
//! stereo use `opus::Decoder`; the multistream variants use `opus::MSDecoder`,
//! which is libopus' own `opus_multistream_decode` and takes the stream count,
//! the coupled-stream count and the channel map from
//! `OpusMultiStreamParameters`. That is one decoder state per libopus stream
//! (stereo for the coupled streams, mono for the rest) with libopus doing the
//! interleaving, so it is the same layout the console uses. `DecodedDataSize`
//! is the header plus payload in bytes and `DecodedSampleCount` is libopus'
//! returned sample count per channel, exactly like the hardware service. When
//! the layout is unusable or a stream fails, the decoder reports the error and
//! returns no samples instead of emitting wrong audio.
//!
//! # Work buffer
//!
//! `hwopus` work-buffer queries report the size of the libopus decoder state:
//! `opus_decoder_get_size` for mono/stereo and
//! `opus_multistream_decoder_get_size` for the multistream layout. That is the
//! real state size of the decoder this service creates, it grows with the
//! channel/stream count as the hardware's does, and the guest page-aligns it
//! before creating the transfer memory. When the requested layout is unusable
//! (or the build has no Opus backend) the query falls back to a documented
//! 0x8000 rather than returning a size that cannot fit a stereo decoder.
//!
//! # Invalid packets
//!
//! A packet whose framing is broken or that libopus rejects is reported as a
//! successful decode with zero samples: `DecodedDataSize` is the number of
//! packet bytes that were consumed (the 8-byte header plus the declared
//! payload, or all available bytes when the header is truncated) and
//! `DecodedSampleCount` is 0. No PCM is written. The SDK uses the sample count
//! to know how much audio it got, so a zero-sample reply is a valid empty
//! frame and keeps the guest's ring accounting consistent; returning an
//! invented error code would be less faithful than the hardware's own result.

use std::sync::Mutex;

pub(crate) const HEADER_SIZE: usize = 8;
/// The longest Opus packet is 120 ms, i.e. 5760 samples per channel at 48 kHz.
#[cfg(feature = "opus")]
const MAX_SAMPLES: usize = 5760;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "opus"), allow(dead_code))]
pub(crate) enum DecodeError {
    /// The input buffer does not hold a complete header plus payload.
    InvalidLength,
    /// libopus rejected the packet.
    InvalidPacket,
    /// The requested sample rate is not one of the Opus rates.
    UnsupportedRate,
    /// A plain mono/stereo decoder was asked for more than two channels.
    UnsupportedChannels,
    /// `OpusMultiStreamParameters` describes an unusable stream layout.
    InvalidLayout,
    /// libopus could not create the decoder, or the build has no Opus backend.
    DecoderUnavailable,
}

impl DecodeError {
    pub(crate) fn label(self) -> &'static str {
        match self {
            DecodeError::InvalidLength => "invalid packet length",
            DecodeError::InvalidPacket => "invalid Opus packet",
            DecodeError::UnsupportedRate => "unsupported sample rate",
            DecodeError::UnsupportedChannels => "multistream Opus needs the multistream decoder",
            DecodeError::InvalidLayout => "invalid multistream layout",
            DecodeError::DecoderUnavailable => "libopus decoder unavailable",
        }
    }
}

pub(crate) struct Decoded {
    pub consumed: u32,
    pub samples: u32,
    pub pcm: Vec<i16>,
    pub error: Option<DecodeError>,
}

impl Decoded {
    fn empty(consumed: u32, error: DecodeError) -> Decoded {
        Decoded {
            consumed,
            samples: 0,
            pcm: Vec::new(),
            error: Some(error),
        }
    }
}

#[cfg(feature = "opus")]
enum Backend {
    Single(opus::Decoder),
    Multi(opus::MSDecoder),
}

struct State {
    #[cfg(feature = "opus")]
    backend: Option<Backend>,
    failure: Option<DecodeError>,
}

pub(crate) struct OpusDecoder {
    #[cfg_attr(not(feature = "opus"), allow(dead_code))]
    channels: u32,
    state: Mutex<State>,
}

fn valid_rate(rate: u32) -> bool {
    matches!(rate, 8000 | 12_000 | 16_000 | 24_000 | 48_000)
}

/// Size reported when no usable libopus state can be sized for the request.
/// It is the value the service used before the exact queries existed and is
/// larger than a stereo `OpusDecoder`.
pub(crate) const FALLBACK_WORK_BUFFER_SIZE: u32 = 0x8000;

/// `GetWorkBufferSize`: bytes of libopus decoder state for a mono/stereo
/// decoder. More than two channels clamps to the stereo state because the
/// plain decoder cannot be opened for more; the multistream query covers the
/// surround layouts. An unusable rate yields the documented fallback.
pub(crate) fn work_buffer_size(sample_rate: u32, channels: u32) -> u32 {
    if !valid_rate(sample_rate) {
        return FALLBACK_WORK_BUFFER_SIZE;
    }
    #[cfg(feature = "opus")]
    {
        // SAFETY: `opus_decoder_get_size` is a pure size query; it takes no
        // pointers and only returns the struct size (or a negative error).
        let size = unsafe { opusic_sys::opus_decoder_get_size(channels.clamp(1, 2) as i32) };
        if size > 0 {
            return size as u32;
        }
    }
    let _ = channels;
    FALLBACK_WORK_BUFFER_SIZE
}

/// `GetWorkBufferSizeForMultiStream`: bytes of libopus multistream decoder
/// state for the same layout [`OpusDecoder::new_multistream`] would create.
/// An unusable layout yields the documented fallback.
pub(crate) fn work_buffer_size_multistream(
    sample_rate: u32,
    channels: u32,
    streams: u32,
    coupled: u32,
    mapping: &[u8],
) -> u32 {
    if !valid_rate(sample_rate) || !multistream_layout_valid(channels, streams, coupled, mapping) {
        return FALLBACK_WORK_BUFFER_SIZE;
    }
    #[cfg(feature = "opus")]
    {
        // SAFETY: `opus_multistream_decoder_get_size` is a pure size query; it
        // takes two integers and returns the struct size (or a negative error).
        let size = unsafe {
            opusic_sys::opus_multistream_decoder_get_size(streams as i32, coupled as i32)
        };
        if size > 0 {
            return size as u32;
        }
    }
    FALLBACK_WORK_BUFFER_SIZE
}

fn multistream_layout_valid(channels: u32, streams: u32, coupled: u32, mapping: &[u8]) -> bool {
    let channels = channels as usize;
    let streams = streams as usize;
    let coupled = coupled as usize;
    channels > 0
        && channels <= 255
        && streams > 0
        && streams <= 255
        && coupled <= streams
        && streams + coupled <= 255
        && mapping.len() >= channels
        && mapping[..channels]
            .iter()
            .all(|&m| m == 255 || usize::from(m) < streams + coupled)
}

impl OpusDecoder {
    pub(crate) fn new(sample_rate: u32, channels: u32) -> OpusDecoder {
        let channels = channels.max(1);
        let state = Self::build(sample_rate, channels);
        OpusDecoder {
            channels,
            state: Mutex::new(state),
        }
    }

    /// Open a multistream decoder with the layout from
    /// `OpusMultiStreamParameters`. `mapping` must hold `channels` entries;
    /// `255` mutes a channel, as libopus documents.
    pub(crate) fn new_multistream(
        sample_rate: u32,
        channels: u32,
        streams: u32,
        coupled: u32,
        mapping: &[u8],
    ) -> OpusDecoder {
        let channels = channels.max(1);
        let state = Self::build_multistream(sample_rate, channels, streams, coupled, mapping);
        OpusDecoder {
            channels,
            state: Mutex::new(state),
        }
    }

    fn build(sample_rate: u32, channels: u32) -> State {
        if !valid_rate(sample_rate) {
            return State {
                #[cfg(feature = "opus")]
                backend: None,
                failure: Some(DecodeError::UnsupportedRate),
            };
        }
        if channels > 2 {
            return State {
                #[cfg(feature = "opus")]
                backend: None,
                failure: Some(DecodeError::UnsupportedChannels),
            };
        }
        #[cfg(feature = "opus")]
        {
            let layout = if channels == 1 {
                opus::Channels::Mono
            } else {
                opus::Channels::Stereo
            };
            match opus::Decoder::new(sample_rate, layout) {
                Ok(decoder) => State {
                    backend: Some(Backend::Single(decoder)),
                    failure: None,
                },
                Err(_) => State {
                    backend: None,
                    failure: Some(DecodeError::DecoderUnavailable),
                },
            }
        }
        #[cfg(not(feature = "opus"))]
        {
            State {
                failure: Some(DecodeError::DecoderUnavailable),
            }
        }
    }

    fn build_multistream(
        sample_rate: u32,
        channels: u32,
        streams: u32,
        coupled: u32,
        mapping: &[u8],
    ) -> State {
        let failure = |error| State {
            #[cfg(feature = "opus")]
            backend: None,
            failure: Some(error),
        };
        if !valid_rate(sample_rate) {
            return failure(DecodeError::UnsupportedRate);
        }
        let channels = channels as usize;
        let streams = streams as usize;
        let coupled = coupled as usize;
        if !multistream_layout_valid(channels as u32, streams as u32, coupled as u32, mapping) {
            return failure(DecodeError::InvalidLayout);
        }
        let mapping = &mapping[..channels];
        #[cfg(feature = "opus")]
        {
            let streams = streams as u8;
            let coupled = coupled as u8;
            match opus::MSDecoder::new(sample_rate, streams, coupled, mapping) {
                Ok(decoder) => State {
                    backend: Some(Backend::Multi(decoder)),
                    failure: None,
                },
                Err(_) => failure(DecodeError::InvalidLayout),
            }
        }
        #[cfg(not(feature = "opus"))]
        {
            let _ = (streams, coupled);
            failure(DecodeError::DecoderUnavailable)
        }
    }

    /// The reason the decoder cannot work, if any.
    pub(crate) fn failure(&self) -> Option<DecodeError> {
        self.state.lock().ok().and_then(|s| s.failure)
    }

    pub(crate) fn set_context(&self) {
        #[cfg(feature = "opus")]
        if let Ok(mut state) = self.state.lock()
            && let Some(backend) = state.backend.as_mut()
        {
            match backend {
                Backend::Single(decoder) => {
                    let _ = decoder.reset_state();
                }
                Backend::Multi(decoder) => {
                    let _ = decoder.reset_state();
                }
            }
        }
    }

    pub(crate) fn decode(&self, data: &[u8], reset: bool) -> Decoded {
        if data.len() < HEADER_SIZE {
            return Decoded::empty(data.len() as u32, DecodeError::InvalidLength);
        }
        let length = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let total = HEADER_SIZE + length;
        if total > data.len() {
            return Decoded::empty(data.len() as u32, DecodeError::InvalidLength);
        }
        let payload = &data[HEADER_SIZE..total];
        #[allow(unused_mut)]
        let Ok(mut state) = self.state.lock() else {
            return Decoded::empty(total as u32, DecodeError::DecoderUnavailable);
        };
        if let Some(error) = state.failure {
            return Decoded::empty(total as u32, error);
        }
        #[cfg(feature = "opus")]
        {
            let Some(backend) = state.backend.as_mut() else {
                return Decoded::empty(total as u32, DecodeError::DecoderUnavailable);
            };
            let result = match backend {
                Backend::Single(decoder) => {
                    Self::decode_single(decoder, payload, reset, self.channels)
                }
                Backend::Multi(decoder) => {
                    Self::decode_multi(decoder, payload, reset, self.channels)
                }
            };
            match result {
                Ok((pcm, samples)) => Decoded {
                    consumed: total as u32,
                    samples: samples as u32,
                    pcm,
                    error: None,
                },
                Err(error) => Decoded::empty(total as u32, error),
            }
        }
        #[cfg(not(feature = "opus"))]
        {
            let _ = (payload, reset, state);
            Decoded::empty(total as u32, DecodeError::DecoderUnavailable)
        }
    }

    #[cfg(feature = "opus")]
    fn decode_single(
        decoder: &mut opus::Decoder,
        payload: &[u8],
        reset: bool,
        channels: u32,
    ) -> Result<(Vec<i16>, usize), DecodeError> {
        let per_channel = match decoder.get_nb_samples(payload) {
            Ok(n) if n > 0 => n,
            _ => return Err(DecodeError::InvalidPacket),
        };
        if reset {
            let _ = decoder.reset_state();
        }
        let mut pcm = vec![0i16; per_channel * channels as usize];
        match decoder.decode(payload, &mut pcm, false) {
            Ok(samples) => {
                pcm.truncate(samples * channels as usize);
                Ok((pcm, samples))
            }
            Err(error) => Err(map_error(error)),
        }
    }

    #[cfg(feature = "opus")]
    fn decode_multi(
        decoder: &mut opus::MSDecoder,
        payload: &[u8],
        reset: bool,
        channels: u32,
    ) -> Result<(Vec<i16>, usize), DecodeError> {
        if reset {
            let _ = decoder.reset_state();
        }
        let mut pcm = vec![0i16; MAX_SAMPLES * channels as usize];
        match decoder.decode(payload, &mut pcm, false) {
            Ok(samples) if samples > 0 => {
                pcm.truncate(samples * channels as usize);
                Ok((pcm, samples))
            }
            Ok(_) => Err(DecodeError::InvalidPacket),
            Err(error) => Err(map_error(error)),
        }
    }
}

#[cfg(feature = "opus")]
fn map_error(error: opus::Error) -> DecodeError {
    match error.code() {
        opus::ErrorCode::InvalidPacket => DecodeError::InvalidPacket,
        opus::ErrorCode::BufferTooSmall | opus::ErrorCode::BadArg => DecodeError::InvalidLength,
        _ => DecodeError::InvalidPacket,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_packets_report_the_available_length() {
        let decoder = OpusDecoder::new(48_000, 2);
        let d = decoder.decode(&[0, 0, 0], false);
        assert_eq!(d.consumed, 3);
        assert_eq!(d.samples, 0);
        assert_eq!(d.error, Some(DecodeError::InvalidLength));

        let d = decoder.decode(&[0, 0, 0, 8, 0, 0, 0, 0, 1], false);
        assert_eq!(d.consumed, 9);
        assert_eq!(d.error, Some(DecodeError::InvalidLength));
    }

    #[test]
    fn multistream_and_odd_rates_are_labelled() {
        let decoder = OpusDecoder::new(48_000, 6);
        assert_eq!(decoder.failure(), Some(DecodeError::UnsupportedChannels));
        let decoder = OpusDecoder::new(44_100, 2);
        assert_eq!(decoder.failure(), Some(DecodeError::UnsupportedRate));
    }

    #[test]
    fn a_bad_multistream_layout_is_rejected() {
        let decoder = OpusDecoder::new_multistream(48_000, 6, 0, 0, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(decoder.failure(), Some(DecodeError::InvalidLayout));
        let decoder = OpusDecoder::new_multistream(48_000, 6, 4, 2, &[0, 1, 2, 3, 4, 6]);
        assert_eq!(decoder.failure(), Some(DecodeError::InvalidLayout));
        let decoder = OpusDecoder::new_multistream(48_000, 6, 4, 2, &[0, 1]);
        assert_eq!(decoder.failure(), Some(DecodeError::InvalidLayout));
        let decoder = OpusDecoder::new_multistream(44_100, 6, 4, 2, &[0, 4, 1, 2, 3, 5]);
        assert_eq!(decoder.failure(), Some(DecodeError::UnsupportedRate));
    }

    #[cfg(feature = "opus")]
    #[test]
    fn the_work_buffer_is_the_libopus_state_size() {
        let mono = work_buffer_size(48_000, 1);
        let stereo = work_buffer_size(48_000, 2);
        assert!(mono >= 4096, "mono state is {mono} bytes");
        assert!(stereo >= 4096, "stereo state is {stereo} bytes");
        assert!(mono < 1 << 20 && stereo < 1 << 20);
        // More than two plain channels is still sized as stereo; the plain
        // decoder cannot open them, but the guest must get a usable size.
        assert_eq!(work_buffer_size(48_000, 6), stereo);
        // An unusable rate reports the documented fallback.
        assert_eq!(work_buffer_size(44_100, 2), FALLBACK_WORK_BUFFER_SIZE);
        assert_eq!(work_buffer_size(48_000, 0), work_buffer_size(48_000, 1));

        let mapping = [0u8, 4, 1, 2, 3, 5];
        let surround = work_buffer_size_multistream(48_000, 6, 4, 2, &mapping);
        assert!(
            surround > stereo,
            "6-channel multistream state {surround} must exceed stereo {stereo}"
        );
        assert!(surround < 4 << 20);
        assert_eq!(
            work_buffer_size_multistream(48_000, 6, 0, 0, &mapping),
            FALLBACK_WORK_BUFFER_SIZE
        );
        assert_eq!(
            work_buffer_size_multistream(48_000, 6, 4, 2, &[0, 1]),
            FALLBACK_WORK_BUFFER_SIZE
        );
        assert_eq!(
            work_buffer_size_multistream(44_100, 6, 4, 2, &mapping),
            FALLBACK_WORK_BUFFER_SIZE
        );
    }

    #[cfg(feature = "opus")]
    fn header(payload: &[u8]) -> Vec<u8> {
        let mut data = Vec::with_capacity(HEADER_SIZE + payload.len());
        data.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(payload);
        data
    }

    #[cfg(feature = "opus")]
    fn tone(samples: usize, channels: usize, amplitude: f32) -> Vec<f32> {
        let mut pcm = vec![0.0f32; samples * channels];
        for i in 0..samples {
            let v = (std::f32::consts::TAU * 220.0 * i as f32 / 48_000.0).sin() * amplitude;
            for c in 0..channels {
                pcm[i * channels + c] = v;
            }
        }
        pcm
    }

    #[cfg(feature = "opus")]
    #[test]
    fn encoded_tone_round_trips_through_the_packet_format() {
        let mut encoder =
            opus::Encoder::new(48_000, opus::Channels::Stereo, opus::Application::Audio)
                .expect("encoder");
        let mut packets = Vec::new();
        for _ in 0..4 {
            let pcm = tone(960, 2, 0.5);
            let encoded = encoder.encode_vec_float(&pcm, 4000).expect("encode");
            assert!(!encoded.is_empty());
            packets.push(header(&encoded));
        }
        let decoder = OpusDecoder::new(48_000, 2);
        assert_eq!(decoder.failure(), None);
        let mut total = 0;
        let mut peak = 0i16;
        for packet in &packets {
            let decoded = decoder.decode(packet, false);
            assert_eq!(decoded.error, None);
            assert_eq!(decoded.consumed as usize, packet.len());
            assert_eq!(decoded.samples, 960);
            assert_eq!(decoded.pcm.len(), 960 * 2);
            assert!(decoded.pcm.iter().all(|v| *v != i16::MIN));
            total += decoded.samples;
            peak = peak.max(decoded.pcm.iter().map(|v| v.abs()).max().unwrap_or(0));
        }
        assert_eq!(total, 4 * 960);
        assert!(peak > 2000, "decoded tone must not be silent: {peak}");
    }

    #[cfg(feature = "opus")]
    #[test]
    fn reset_is_accepted_and_silence_stays_silent() {
        let mut encoder =
            opus::Encoder::new(48_000, opus::Channels::Mono, opus::Application::Audio)
                .expect("encoder");
        let decoded = OpusDecoder::new(48_000, 1);
        for i in 0..3 {
            let pcm = vec![0.0f32; 960];
            let encoded = encoder.encode_vec_float(&pcm, 4000).expect("encode");
            let d = decoded.decode(&header(&encoded), i == 1);
            assert_eq!(d.error, None);
            assert_eq!(d.samples, 960);
            assert!(
                d.pcm.iter().all(|v| v.abs() < 8),
                "silence leaked: {:?}",
                d.pcm
            );
        }
    }

    #[cfg(feature = "opus")]
    #[test]
    fn a_six_channel_multistream_packet_interleaves_every_stream() {
        // Standard 5.1 layout (RFC 7845 family 1): four streams, two coupled,
        // mapping [0, 4, 1, 2, 3, 5]. Distinct amplitudes identify each channel
        // in the decoded result: [s0.L, s2, s0.R, s1.L, s1.R, s3].
        let amplitudes = [0.8f32, 0.6, 0.4, 0.2, 0.1, 0.05];
        let mapping = [0u8, 4, 1, 2, 3, 5];
        let mut encoder = opus::MSEncoder::new(48_000, 4, 2, &mapping, opus::Application::Audio)
            .expect("multistream encoder");
        let decoder = OpusDecoder::new_multistream(48_000, 6, 4, 2, &mapping);
        assert_eq!(decoder.failure(), None);

        let mut sums = [0f64; 6];
        let mut count = 0usize;
        for block in 0..6 {
            let mut pcm = vec![0.0f32; 960 * 6];
            for i in 0..960 {
                let v = (std::f32::consts::TAU * 220.0 * i as f32 / 48_000.0).sin();
                for c in 0..6 {
                    pcm[i * 6 + c] = v * amplitudes[c];
                }
            }
            let encoded = encoder.encode_vec_float(&pcm, 4000).expect("encode");
            let decoded = decoder.decode(&header(&encoded), false);
            assert_eq!(decoded.error, None, "block {block}");
            assert_eq!(decoded.samples, 960);
            assert_eq!(decoded.pcm.len(), 960 * 6);
            if block > 0 {
                for frame in 0..960 {
                    for (channel, sum) in sums.iter_mut().enumerate() {
                        let v = f64::from(decoded.pcm[frame * 6 + channel]) / 32768.0;
                        *sum += v * v;
                    }
                }
                count += 960;
            }
        }
        let rms: Vec<f64> = sums.iter().map(|sum| (sum / count as f64).sqrt()).collect();
        for i in 0..6 {
            assert!(rms[i] > 1.0e-3, "channel {i} is silent: {rms:?}");
        }
        for i in 0..5 {
            assert!(
                rms[i] > rms[i + 1] * 1.3,
                "channel order does not match the mapping: {rms:?}"
            );
        }
    }

    #[cfg(feature = "opus")]
    #[test]
    fn a_truncated_multistream_packet_reports_silence() {
        let mapping = [0u8, 4, 1, 2, 3, 5];
        let mut encoder = opus::MSEncoder::new(48_000, 4, 2, &mapping, opus::Application::Audio)
            .expect("multistream encoder");
        let mut pcm = vec![0.0f32; 960 * 6];
        for (i, sample) in pcm.iter_mut().enumerate() {
            *sample = (i % 97) as f32 / 200.0;
        }
        let encoded = encoder.encode_vec_float(&pcm, 4000).expect("encode");
        let decoder = OpusDecoder::new_multistream(48_000, 6, 4, 2, &mapping);
        let mut packet = encoded.clone();
        packet.truncate(packet.len() / 2);
        let decoded = decoder.decode(&header(&packet), false);
        assert_eq!(decoded.samples, 0);
        assert!(decoded.pcm.is_empty());
        assert_eq!(decoded.error, Some(DecodeError::InvalidPacket));
    }

    #[cfg(feature = "opus")]
    #[test]
    fn a_multichannel_single_decoder_reports_silence_with_a_reason() {
        let decoder = OpusDecoder::new(48_000, 6);
        let mut encoder =
            opus::Encoder::new(48_000, opus::Channels::Stereo, opus::Application::Audio)
                .expect("encoder");
        let encoded = encoder
            .encode_vec_float(&tone(960, 2, 0.5), 4000)
            .expect("encode");
        let decoded = decoder.decode(&header(&encoded), false);
        assert_eq!(decoded.error, Some(DecodeError::UnsupportedChannels));
        assert_eq!(decoded.samples, 0);
        assert!(decoded.pcm.is_empty());
    }
}
