from __future__ import annotations

from pathlib import Path
from typing import Callable

from .auto_trace_service import AutoTraceService
from .types import ProcessResult


class BatchProcessor:
    def __init__(self, service: AutoTraceService):
        self.service = service

    def run(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
        export_formats: tuple[str, ...] | list[str] = ("svg",),
        progress_callback: Callable[[int, int, Path], None] | None = None,
    ) -> list[ProcessResult]:
        total = len(input_paths)
        if total == 0:
            return []

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        results: list[ProcessResult] = []
        for index, value in enumerate(input_paths, start=1):
            path = Path(value)
            if progress_callback is not None:
                progress_callback(index, total, path)

            result = self.service.run(
                input_path=path,
                output_dir=output_root,
                export_formats=export_formats,
                export_basename=path.stem,
            )
            results.append(result)
        return results
