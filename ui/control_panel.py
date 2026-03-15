from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.types import PipelineConfig


class TraceControlPanel(QWidget):
    settingsChanged = Signal()
    zoomChanged = Signal(float)
    overlayChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        self.setMinimumWidth(320)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        self.status_label = QLabel("Preset: Auto")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-weight: 600; color: #1f2937;")
        layout.addWidget(self.status_label)

        self.analysis_label = QLabel("Analysis not available yet.")
        self.analysis_label.setWordWrap(True)
        self.analysis_label.setStyleSheet("color: #4b5563;")
        layout.addWidget(self.analysis_label)

        mode_group = QGroupBox("Trace Mode")
        mode_form = QFormLayout()
        self.preset_combo = QComboBox()
        self._fill_combo(
            self.preset_combo,
            [
                ("Auto", "auto"),
                ("Logo", "logo"),
                ("Illustration", "illustration"),
                ("Text / Typo", "text_typo"),
                ("DTF / Screen Print", "dtf_screen_print"),
                ("Sticker / Cutting Plotter", "sticker_cutting"),
                ("Photo", "photo"),
            ],
        )
        self.quality_combo = QComboBox()
        self._fill_combo(
            self.quality_combo,
            [
                ("Balanced", "balanced"),
                ("High Quality", "high_quality"),
            ],
        )
        self.background_combo = QComboBox()
        self._fill_combo(
            self.background_combo,
            [
                ("Transparent", "transparent"),
                ("Keep White", "keep_white"),
                ("Drop White", "drop_white"),
            ],
        )
        self.colors_combo = QComboBox()
        self._fill_combo(self.colors_combo, [("1", 1), ("2", 2), ("4", 4), ("6", 6), ("8", 8)])
        mode_form.addRow("Preset", self.preset_combo)
        mode_form.addRow("Quality", self.quality_combo)
        mode_form.addRow("Background", self.background_combo)
        mode_form.addRow("Colors", self.colors_combo)
        mode_group.setLayout(mode_form)
        layout.addWidget(mode_group)

        controls_group = QGroupBox("Pro Controls")
        controls_layout = QGridLayout()
        self.detail_slider = self._build_slider("Detail", 0, 100, 70)
        self.smooth_slider = self._build_slider("Smoothness", 0, 100, 35)
        self.corner_slider = self._build_slider("Corners", 0, 100, 75)
        self.despeckle_slider = self._build_slider("Despeckle", 0, 100, 35)
        self.min_shape_spin = QSpinBox()
        self.min_shape_spin.setRange(1, 2000)
        self.min_shape_spin.setValue(18)
        self.cutline_offset_spin = QSpinBox()
        self.cutline_offset_spin.setRange(0, 128)
        self.cutline_offset_spin.setValue(8)
        self.outline_slider = self._build_slider("Outline", 0, 100, 55)

        controls_layout.addWidget(self.detail_slider["container"], 0, 0)
        controls_layout.addWidget(self.smooth_slider["container"], 1, 0)
        controls_layout.addWidget(self.corner_slider["container"], 2, 0)
        controls_layout.addWidget(self.despeckle_slider["container"], 3, 0)

        min_shape_row = QWidget()
        min_shape_layout = QHBoxLayout()
        min_shape_layout.setContentsMargins(0, 0, 0, 0)
        min_shape_layout.addWidget(QLabel("Min Shape Size"))
        min_shape_layout.addWidget(self.min_shape_spin)
        min_shape_row.setLayout(min_shape_layout)
        controls_layout.addWidget(min_shape_row, 4, 0)

        cutline_row = QWidget()
        cutline_layout = QHBoxLayout()
        cutline_layout.setContentsMargins(0, 0, 0, 0)
        cutline_layout.addWidget(QLabel("Cutline Offset"))
        cutline_layout.addWidget(self.cutline_offset_spin)
        cutline_row.setLayout(cutline_layout)
        controls_layout.addWidget(cutline_row, 5, 0)

        controls_layout.addWidget(self.outline_slider["container"], 6, 0)
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        controlnet_group = QGroupBox("AI Outline (Photo Pipeline)")
        controlnet_layout = QFormLayout()
        self.controlnet_enable_check = QCheckBox("Enable ControlNet")
        self.controlnet_url_edit = QLineEdit("http://127.0.0.1:7860")
        self.controlnet_preprocessor_combo = QComboBox()
        self._fill_combo(
            self.controlnet_preprocessor_combo,
            [
                ("Lineart", "lineart"),
                ("Canny", "canny"),
            ],
        )
        self.controlnet_model_label = QLabel("control_v11p_sd15_lineart")
        self.controlnet_model_label.setWordWrap(True)
        self.controlnet_note_label = QLabel(
            "Hanya dipakai untuk scan/foto kaos pada pipeline photo. Artwork transparan dan raster artwork tetap memakai pipeline lokal."
        )
        self.controlnet_note_label.setWordWrap(True)
        self.controlnet_note_label.setStyleSheet("color: #4b5563;")
        controlnet_layout.addRow(self.controlnet_enable_check)
        controlnet_layout.addRow("A1111 URL", self.controlnet_url_edit)
        controlnet_layout.addRow("Preprocessor", self.controlnet_preprocessor_combo)
        controlnet_layout.addRow("Model", self.controlnet_model_label)
        controlnet_layout.addRow(self.controlnet_note_label)
        controlnet_group.setLayout(controlnet_layout)
        layout.addWidget(controlnet_group)

        option_group = QGroupBox("Options")
        option_layout = QVBoxLayout()
        self.small_boost_check = QCheckBox("Small Image Boost")
        self.small_boost_check.setChecked(True)
        self.ignore_white_check = QCheckBox("Ignore White Shapes")
        self.ignore_white_check.setChecked(False)
        self.overlay_check = QCheckBox("Overlay Segmentation")
        self.overlay_check.setChecked(True)
        option_layout.addWidget(self.small_boost_check)
        option_layout.addWidget(self.ignore_white_check)
        option_layout.addWidget(self.overlay_check)
        option_group.setLayout(option_layout)
        layout.addWidget(option_group)

        zoom_group = QGroupBox("Preview Zoom")
        zoom_layout = QVBoxLayout()
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom"))
        self.zoom_value_label = QLabel("100%")
        self.zoom_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        zoom_row.addWidget(self.zoom_value_label)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(40, 240)
        self.zoom_slider.setValue(100)
        zoom_layout.addLayout(zoom_row)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_group.setLayout(zoom_layout)
        layout.addWidget(zoom_group)

        layout.addStretch(1)
        self.setLayout(layout)

        widgets = [
            self.preset_combo,
            self.quality_combo,
            self.background_combo,
            self.colors_combo,
            self.min_shape_spin,
            self.cutline_offset_spin,
            self.small_boost_check,
            self.ignore_white_check,
            self.controlnet_enable_check,
            self.controlnet_preprocessor_combo,
        ]
        widgets.extend(
            [
                self.detail_slider["slider"],
                self.smooth_slider["slider"],
                self.corner_slider["slider"],
                self.despeckle_slider["slider"],
                self.outline_slider["slider"],
            ]
        )
        for widget in widgets:
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._emit_settings_changed)
            elif hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._emit_settings_changed)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self._emit_settings_changed)

        self.controlnet_url_edit.textChanged.connect(self._emit_settings_changed)
        self.controlnet_preprocessor_combo.currentIndexChanged.connect(self._refresh_controlnet_model_label)
        self.zoom_slider.valueChanged.connect(self._emit_zoom_changed)
        self.overlay_check.toggled.connect(self.overlayChanged.emit)
        self._refresh_controlnet_model_label()

    def apply_to_config(self, config: PipelineConfig) -> None:
        config.preset = str(self.preset_combo.currentData())
        config.quality_mode = str(self.quality_combo.currentData())
        config.background_mode = str(self.background_combo.currentData())
        config.color_count = int(self.colors_combo.currentData())
        config.settings.detail = int(self.detail_slider["slider"].value())
        config.settings.smoothness = int(self.smooth_slider["slider"].value())
        config.settings.corners = int(self.corner_slider["slider"].value())
        config.settings.despeckle = int(self.despeckle_slider["slider"].value())
        config.settings.min_shape_area = int(self.min_shape_spin.value())
        config.settings.cutline_offset = int(self.cutline_offset_spin.value())
        config.settings.small_image_boost = bool(self.small_boost_check.isChecked())
        config.settings.ignore_white = bool(self.ignore_white_check.isChecked())
        config.settings.outline_strength = int(self.outline_slider["slider"].value())
        config.controlnet.enabled = bool(self.controlnet_enable_check.isChecked())
        config.controlnet.base_url = str(self.controlnet_url_edit.text()).strip()
        config.controlnet.preprocessor = str(self.controlnet_preprocessor_combo.currentData())

    def sync_from_config(self, config: PipelineConfig) -> None:
        preset_value = config.validated_preset()
        if preset_value == "sablon":
            preset_value = "dtf_screen_print"
        self._set_combo_value(self.preset_combo, preset_value)
        self._set_combo_value(self.quality_combo, config.validated_quality_mode())
        self._set_combo_value(self.background_combo, config.validated_background_mode())
        self._set_combo_value(self.colors_combo, config.validated_color_count())
        settings = config.settings.clamped()
        self.detail_slider["slider"].setValue(settings.detail)
        self.smooth_slider["slider"].setValue(settings.smoothness)
        self.corner_slider["slider"].setValue(settings.corners)
        self.despeckle_slider["slider"].setValue(settings.despeckle)
        self.min_shape_spin.setValue(settings.min_shape_area)
        self.cutline_offset_spin.setValue(settings.cutline_offset)
        self.small_boost_check.setChecked(settings.small_image_boost)
        self.ignore_white_check.setChecked(settings.ignore_white)
        self.outline_slider["slider"].setValue(settings.outline_strength)
        self.controlnet_enable_check.setChecked(bool(config.controlnet.enabled))
        self.controlnet_url_edit.setText(config.controlnet.normalized_base_url())
        self._set_combo_value(self.controlnet_preprocessor_combo, config.controlnet.validated_preprocessor())
        self._refresh_controlnet_model_label()

    def update_result_summary(self, metadata: dict[str, object]) -> None:
        preset = str(metadata.get("presetUsed", "auto")).replace("_", " ").title()
        quality = str(metadata.get("qualityMode", "balanced")).replace("_", " ").title()
        pipeline = str(metadata.get("pipelineMode", "photo")).replace("_", " ").title()
        controlnet = metadata.get("controlnet", {})
        outline_backend = str(metadata.get("outlineBackend", "classic")).strip().lower()
        outline_label = "Classic"
        if outline_backend == "controlnet" and isinstance(controlnet, dict):
            preprocessor = str(controlnet.get("preprocessor", "lineart")).replace("_", " ").title()
            outline_label = f"ControlNet {preprocessor}"
        self.status_label.setText(f"Preset: {preset} | Pipeline: {pipeline} | Quality: {quality} | Outline: {outline_label}")

        analysis = metadata.get("analysis", {})
        if not isinstance(analysis, dict):
            self.analysis_label.setText("Analysis unavailable.")
            return

        scores = analysis.get("scores", {})
        if isinstance(scores, dict) and scores:
            ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            summary = ", ".join(f"{key}:{float(value):.2f}" for key, value in ordered)
        else:
            summary = "No score data"
        background = str(analysis.get("background_type", "unknown")).title()
        complexity = float(analysis.get("image_complexity", 0.0))
        self.analysis_label.setText(f"Scores {summary}\nBackground {background} | Complexity {complexity:.2f}")

    def overlay_enabled(self) -> bool:
        return self.overlay_check.isChecked()

    def _emit_zoom_changed(self, value: int) -> None:
        percent = max(40, min(240, int(value)))
        self.zoom_value_label.setText(f"{percent}%")
        self.zoomChanged.emit(percent / 100.0)

    def _emit_settings_changed(self, *args) -> None:
        del args
        self.settingsChanged.emit()

    def _refresh_controlnet_model_label(self, *args) -> None:
        del args
        preprocessor = str(self.controlnet_preprocessor_combo.currentData() or "lineart")
        mapping = {
            "lineart": "control_v11p_sd15_lineart",
            "canny": "control_v11p_sd15_canny",
        }
        self.controlnet_model_label.setText(mapping.get(preprocessor, "control_v11p_sd15_lineart"))

    def _build_slider(self, title: str, minimum: int, maximum: int, value: int) -> dict[str, object]:
        container = QFrame()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        label = QLabel(title)
        value_label = QLabel(str(value))
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.valueChanged.connect(lambda current, target=value_label: target.setText(str(current)))
        header.addWidget(label)
        header.addWidget(value_label)
        layout.addLayout(header)
        layout.addWidget(slider)
        container.setLayout(layout)
        return {"container": container, "slider": slider, "value_label": value_label}

    def _fill_combo(self, combo: QComboBox, items: list[tuple[str, object]]) -> None:
        for label, data in items:
            combo.addItem(label, userData=data)

    def _set_combo_value(self, combo: QComboBox, data: object) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == data:
                combo.setCurrentIndex(index)
                return
