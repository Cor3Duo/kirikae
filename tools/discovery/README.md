# Discovery loop

Mechanical half of the plan in [`docs/plan/MASTER-PLAN.md`](../../docs/plan/MASTER-PLAN.md):
build a game, run it headless under scripted input, collect artifacts, and turn them
into deduplicated, routed work items. No AI, no network, no game data in the repository.

## Quick start

```bash
# 1. build kirikae once
cargo build --release -p kirikae

# 2. copy and edit the example config
cp tools/discovery/discovery.example.json discovery.json

# 3. one full cycle (build + run + triage)
python tools/discovery/loop.py --config discovery.json

# just triage artifacts you already have
python tools/discovery/triage.py --game smbw \
  --coverage myswitch/build/extracted.coverage.json \
  --log myswitch/build/run2.log
```

Artifacts go to `$KIRIKAE_DISCOVERY_DIR` (default `%LOCALAPPDATA%\kirikae-discovery`),
never into the tree. Each cycle writes:

```
<root>/<game>/<run-id>/
  build.log
  run/<scenario>/{stderr.log, coverage.json, game.profile, saves/}
<root>/<game>/items.jsonl     # merged work items (one per fingerprint)
<root>/<game>/summary.md      # priority table + delta
<root>/last-run.json          # manifest of the cycle
```

## What `loop.py` does

1. `kirikae build <source> -o <out>` (optionally with the previous `.profile`).
2. For every scenario: run `<out>/<name>.exe` with
   `KIRIKAE_HEADLESS=1`, `KIRIKAE_COVERAGE`, `KIRIKAE_PROFILE`, `KIRIKAE_STATS=1`,
   `KIRIKAE_SAVES` under the run directory, and `KIRIKAE_INPUT` when given.
3. Calls `triage.py` with every `coverage.json` and `stderr.log`.

Timeouts kill the process (`timeout`) and the exit code is recorded; a crash is an
artifact, not a failure of the loop.

## Many games: `batch.py`

`batch.py` runs a folder of already extracted games through a compile pass and an
optional boot pass, resuming from `state.jsonl`, and writes `compat.md` / `compat.json`
(games × status × functions × frames × gaps):

```bash
python tools/discovery/batch.py --config tools/discovery/batch.example.json
python tools/discovery/batch.py --config batch.json --limit 10
python tools/discovery/batch.py --config batch.json --compile-only
python tools/discovery/batch.py --config batch.json --profile-loop --profile-seconds 60
```

`--profile-loop` builds through `kirikae build --profile-loop`, which alternates a build, a
bounded headless run and a rebuild with what the run interpreted, stopping when the count
stops falling or a run adds nothing new. It starts each game, so it is opt-in; the per-run bound is
`--profile-seconds` (default `boot_seconds`, at least 10) and the whole command has its own
timeout. The iterations land in `compat.json` and in the matrix's Profile column.

Each game gets `<work>/<game>/{out,build.log,run/}`; `work` defaults to
`%LOCALAPPDATA%/kirikae-discovery/batch`. The matrix is the per-game companion of
`docs/progress`: the dashboard says what the runtime implements, the matrix says which
game got there.

## What `triage.py` does

* Reads `items` from the coverage JSON, keeps everything that is not `implemented`.
* Mines logs for `svcBreak`, unimplemented/missing commands, deadlocks and aborts.
* Fingerprints each item as `sha1(kind|section|interface|key)` and merges it with the
  persistent `items.jsonl` (counts, first/last seen, evidence, runs).
* Routes the item to the source file that most likely implements it
  (`crates/rt/src/services/<family>.rs`, `crates/rt/src/horizon.rs`,
  `crates/maxwell/src/translate.rs`, …).
* Assigns a mechanical priority: `P0` for crashes/boot blockers, `P1` for hot gaps,
  `P2` otherwise.
* **Verifies** a work item when the same run's coverage reports that command, SVC, ioctl
  or shader as `implemented`: the item moves to `verified` with `verified_by` set to the
  run id. Absence alone does not close anything.

Work item fields: `id`, `fingerprint`, `kind`, `section`, `interface`, `key`, `status`,
`count`, `first`, `note`, `evidence`, `files_hint`, `priority`, `state`, `attempts`,
`runs`, `first_seen`, `last_seen`.

`state` starts at `new`. The analysis layer (human or AI) moves it through `triaged`,
`investigating`, `patching`, `testing`, `ready_for_review`, and only a human merges.

## Rules

* Never point `source` at a folder with keys or ROMs you cannot use legally.
* Never copy artifacts into the repository: they contain game frames and saves.
* A gap is only closed with a test that fails before and passes after (see the plan).
