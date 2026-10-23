# Contributing to Kirikae

Thanks for helping! Issues and pull requests are welcome in English or Portuguese.

## Before you start

- Read [LEGAL.md](LEGAL.md). In short: clean-room work only, no GPL code, no leaked or NDA material,
  and no game data of any kind (binaries, shaders, textures, screenshots, dumps, keys, firmware).
- Kirikae only reads decrypted dumps. Changes that add decryption are not accepted.
- Game-specific fixes belong in a plugin (`kirikae build --plugin`), not in the runtime, unless they
  fix how the hardware or the system really behaves.

## Building and testing

```bash
cargo build --release
cargo test --release
python tools/progress.py   # after runtime changes: updates docs/progress and the badges
```

On Windows, set `CARGO_TARGET_DIR` to a short folder if the project lives in a long path.

## Pull requests

- Keep each pull request focused on one change.
- Commit messages start with `feat:`, `fix:` or `chore:`, in English, short and to the point.
- Match the surrounding code; the code base keeps comments to the required ones (`// SAFETY:` on
  `unsafe` blocks and third-party license notices).
- Describe how you tested it: the test that covers it, or what you ran and what you saw.
