"""Production entry point for the CricketVision HTTP API.

Run it from the repository root with the backend directory on the import path:

    python backend/main.py

or, equivalently:

    uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000

The application object is built by `app.api.app.create_app`; tests use that
factory directly with a temporary data root and a deterministic pipeline.
"""

from __future__ import annotations

import os

from app.api.app import create_app

app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("CRICKET_HOST", "127.0.0.1")
    port = int(os.environ.get("CRICKET_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
