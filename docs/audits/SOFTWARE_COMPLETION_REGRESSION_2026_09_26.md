# Software-completion regression record

Date: 2026-09-26

This record covers commit
`035609843c675163c2c4d5549b4838d564f1b873` on
`kerem/software-completion`. It is a repository regression record, not a Unity
or physical-validation substitute.

## Current working tree

- `ruff format --check .`: PASS, 802 files already formatted.
- `ruff check .`: PASS.
- `mypy conrad tests`: PASS, 650 source files.
- `pytest -q`: PASS, 1497 passed, 15 skipped, 125 deselected, 3 expected
  failures in 360.95 s.
- `scripts/secret_scan.py`: PASS, zero problems in 1768 tracked files.

The three expected failures are the immutable I4 negative-result assertions
(two) and the immutable I7 criterion-3 negative-result assertion (one). The 15
skips are eleven fail-closed historical I5 artifact checks, one unavailable
`uv` executable check, one missing captured Unity bundle check, and two
optional public-data downloads. No setup error or unexpected failure occurred.

## Fresh remote clone

A new single-branch clone of `origin/kerem/software-completion` was created at
the exact commit above, without working-tree artifacts or caches from the
development checkout. Using the already-installed dependency environment but
imports and configuration from the clone:

- format, lint, and mypy passed with the same 802/650 counts;
- `pytest -q` passed with the same 1497/15/125/3 result in 354.85 s;
- the secret scan found zero problems in 1768 tracked files; and
- the clone remained clean after verification.

This demonstrates that the normal software suite is reproducible from the
remote repository and does not silently depend on local gate bundles or the
generated Spatial V1.1 controlled-view databases.

## Explicit remaining software environment block

Unity live tests remain deselected or skipped. This Apple-silicon host has no
recoverable Windows player artifact or Wine runtime; the repository, Git
history, GitHub releases, and GitHub Actions artifacts contain no player
binary. Unity Hub 3.21.3 and the exact Apple-silicon editor 6000.5.9f1 changeset
`b57deb96f08d` were installed after the regression run, but batch project
compile exits 198 because no Unity Editor license is active. Spatial V1.1 Unity
parity and all downstream I5 validation/final/formal work therefore remain
unopened, not passed. Unity Hub account sign-in and license activation are
required before a native player can be built and a new platform-specific
parity protocol can execute.

## Post-record Unity completion

The blocker above was subsequently cleared by user license activation. The
exact editor compiled the project, a native macOS player was built, and a
separately declared parity supplement passed 16/16 on fresh Unity seeds. The
result is recorded at commit
`3c9adec495eacb2b50b4bbbab9b37038bad5dda7` and frozen with its exact platform
and player-hash boundary in `../architecture/SPATIAL_V1_1_FREEZE.md`. This
later result does not retroactively add Unity execution to the broad regression
snapshot described above.
