"""FastAPI routers. Each module owns one part of the six-route contract."""

from .analysis import router as analysis_router
from .media import router as media_router
from .system import router as system_router

__all__ = ["analysis_router", "media_router", "system_router"]
