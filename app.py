from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from config import APP_NAME, ORGANIZATION_NAME
from ui_main import create_main_window


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORGANIZATION_NAME)

    window = create_main_window(project_root=Path(__file__).resolve().parent)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
