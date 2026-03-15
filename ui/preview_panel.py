from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget


class PreviewPanel(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: 600; font-size: 13px;")

        self.image_label = QLabel("No Preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("background: #f3f3f3; border: 1px solid #d9d9d9;")

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setStyleSheet("background: #f3f3f3; border: 1px solid #d9d9d9;")

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.scroll_area, stretch=1)
        self.setLayout(layout)

        self._pixmap: QPixmap | None = None
        self._zoom_factor = 1.0

    def clear(self) -> None:
        self._pixmap = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("No Preview")

    def set_image(self, image: np.ndarray) -> None:
        pixmap = numpy_to_pixmap(image)
        self._pixmap = pixmap
        self._apply_scaled_pixmap()

    def set_svg(self, svg_text: str) -> None:
        renderer = QSvgRenderer(svg_text.encode("utf-8"))
        if not renderer.isValid():
            self.image_label.setText("Invalid SVG")
            return

        size = renderer.defaultSize()
        width = max(1, size.width(), self.scroll_area.viewport().width())
        height = max(1, size.height(), self.scroll_area.viewport().height())
        image = QImage(width, height, QImage.Format_ARGB32)
        image.fill(Qt.white)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()

        self._pixmap = QPixmap.fromImage(image)
        self._apply_scaled_pixmap()

    def set_zoom_factor(self, value: float) -> None:
        self._zoom_factor = max(0.2, min(4.0, float(value)))
        if self._pixmap is not None:
            self._apply_scaled_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._pixmap is not None:
            self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            return

        viewport = self.scroll_area.viewport().size()
        fit_width = max(1, viewport.width())
        fit_height = max(1, viewport.height())
        scaled = self._pixmap.scaled(
            int(fit_width * self._zoom_factor),
            int(fit_height * self._zoom_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())
        self.image_label.setText("")


def numpy_to_pixmap(image: np.ndarray) -> QPixmap:
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = rgb.shape[:2]
    bytes_per_line = rgb.strides[0]
    qimage = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)
