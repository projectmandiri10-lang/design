from __future__ import annotations

from pathlib import Path

from ui.main_window import MainWindow


def create_main_window(project_root: Path) -> MainWindow:
    return MainWindow(project_root=project_root)


__all__ = ["MainWindow", "create_main_window"]
