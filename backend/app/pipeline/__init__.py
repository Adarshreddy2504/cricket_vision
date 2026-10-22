"""Analysis orchestration shared by the HTTP server and its tests."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipelineResult:
    """The clean browser document plus the diagnostic document.

    `result` is the only thing served to the frontend. `diagnostics` is written
    beside it for a human investigating a boundary, a missing model or an
    unexpected label; it is never sent to the browser.
    """

    result: dict
    diagnostics: dict = field(default_factory=dict)


def __getattr__(name: str):
    # `AnalysisPipeline` imports the package for `PipelineResult`; exporting it
    # eagerly here would create that cycle. The lazy attribute keeps both the
    # test import and the production import path simple.
    if name == "AnalysisPipeline":
        from .analysis import AnalysisPipeline

        return AnalysisPipeline
    raise AttributeError(name)


__all__ = ["AnalysisPipeline", "PipelineResult"]
