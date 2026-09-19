# Legacy independence evidence

Executed on 2026-09-19 with `python scripts/legacy_independence_check.py --rename`. The legacy repository `C:\Users\Kerem\OneDrive\Documents\ChatGPT\conrad` was renamed away during the run and restored afterwards: HEAD `6bf01ee6ef189145fbc57b65aaed2de46ccea45b`, `git status --porcelain` 0 lines before and after. The clean clone ran from a temporary directory.

```
[OK] uv sync --all-groups --locked (31.9s)  + zipp==4.1.0
[OK] ruff format --check (1.3s) 570 files already formatted
[OK] ruff check (0.5s) All checks passed!
[OK] mypy conrad tests (108.8s) Success: no issues found in 518 source files
[OK] tests (443.4s) 840 passed in 431.38s (0:07:11)
[OK] conrad db migrate (4.0s) database C:\Users\Kerem\AppData\Local\Temp\conrad-v2-clean-rm5aqxro\conrad-v2\artifacts\conrad.sqlite at revision 0001_initial
[OK] conrad doctor (9.0s) RESULT: OK
[OK] simulation smoke (24.4s) }
[OK] replay smoke (24.0s) RESULT: REPRODUCED
[OK] core training smoke (18.2s) }
LEGACY INDEPENDENCE: PASS
```
