from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6_QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QApplication = PySide6_QtWidgets.QApplication

from ui.main_window import MainWindow


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_exposes_prompt_workflow_controls(tmp_path: Path) -> None:
    _app()
    window = MainWindow(project_root=tmp_path)

    assert window.windowTitle() == "AUTO VECTOR SABLON AI"
    assert window.import_button.text() == "Import Image"
    assert window.clean_button.text() == "Clean AI"
    assert window.outline_button.text() == "Detect Outline"
    assert window.reduce_button.text() == "Reduce Colors"
    assert window.vector_button.text() == "Generate Vector"
    assert window.export_button.text() == "Export SVG"
    assert window.console_log.isReadOnly() is True
