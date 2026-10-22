"""
conftest.py — test session setup.

One job: put the import root on `sys.path`.

`backend/` is the import root for both `app.*` and `tests.*`, so tests must
otherwise be run with `backend/` as the working directory. Doing it here means
`pytest` works from the project root as well.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

for _p in (BACKEND_DIR, PROJECT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
