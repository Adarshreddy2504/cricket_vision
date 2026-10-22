"""HTTP surface for the CricketVision backend.

The API is deliberately thin: it validates uploads, owns the process-wide job
registry and serves the clean frontend document. All cricket work happens in
`app.pipeline`, using the tracking, pitch, shot and analytics packages.
"""

from .app import create_app

__all__ = ["create_app"]
