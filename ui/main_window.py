from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from config import APP_NAME, build_default_pipeline_config, build_runtime_config
from core.auto_trace_service import AutoTraceService
from core.batch_processor import BatchProcessor
from core.cutline import build_cutline_paths
from core.image_preprocess import compose_alpha_preview, load_image_data, resize_loaded_image
from core.types import ProcessResult
from export.export_eps import convert_svg_to_eps_pdf
from export.export_svg import compose_cutline_svg_string, export_svg_file
from modules.ai_cleanup import CleanupArtifacts, cleanup_image
from modules.color_reduce import ColorReductionArtifacts, reduce_palette
from modules.edge_detect import OutlineArtifacts, detect_outline
from modules.export_svg import export_timestamped_svg
from modules.preprocess import PreprocessArtifacts, prepare_photo_input
from modules.vectorize import VectorArtifacts, generate_vector_svg

from .control_panel import TraceControlPanel
from .preview_panel import PreviewPanel

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(slots=True)
class WorkflowState:
    input_path: Path | None = None
    original_preview: np.ndarray | None = None
    preprocess: PreprocessArtifacts | None = None
    cleanup: CleanupArtifacts | None = None
    outline: OutlineArtifacts | None = None
    colors: ColorReductionArtifacts | None = None
    vector: VectorArtifacts | None = None
    exported_svg: Path | None = None


class WorkerSignals(QObject):
    started = Signal(str)
    success = Signal(object)
    failed = Signal(str)


class CallableWorker(QRunnable):
    def __init__(self, description: str, callback):
        super().__init__()
        self.description = description
        self.callback = callback
        self.signals = WorkerSignals()

    def run(self) -> None:
        self.signals.started.emit(self.description)
        try:
            result = self.callback()
            self.signals.success.emit(result)
        except Exception:  # pragma: no cover - UI boundary
            self.signals.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = project_root
        self.runtime_config = build_runtime_config()
        self.config = build_default_pipeline_config(project_root)
        self._updating_controls = False
        self.workflow = WorkflowState()
        self.current_image_path: Path | None = None
        self.last_result: ProcessResult | None = None
        self.batch_files: list[Path] = []
        self.thread_pool = QThreadPool.globalInstance()

        self.service = AutoTraceService(self.config)
        self.batch_processor = BatchProcessor(self.service)

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1360, 860)
        self.setAcceptDrops(True)

        self._build_ui()
        self._sync_prompt_controls_from_config()
        self._refresh_runtime_hint()
        self._refresh_button_states()

    def _build_ui(self) -> None:
        toolbar = QToolBar("Advanced Tools")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.auto_trace_action = QAction("Auto Trace", self)
        self.auto_trace_action.triggered.connect(self.run_auto_trace)
        toolbar.addAction(self.auto_trace_action)

        self.export_cutline_action = QAction("Export Cutline", self)
        self.export_cutline_action.triggered.connect(self.export_cutline_svg)
        toolbar.addAction(self.export_cutline_action)

        self.export_vector_action = QAction("Export EPS/PDF", self)
        self.export_vector_action.triggered.connect(self.export_eps_pdf)
        toolbar.addAction(self.export_vector_action)

        toolbar.addSeparator()
        self.add_batch_action = QAction("Add Batch", self)
        self.add_batch_action.triggered.connect(self.add_batch_files)
        toolbar.addAction(self.add_batch_action)

        self.run_batch_action = QAction("Run Batch", self)
        self.run_batch_action.triggered.connect(self.run_batch)
        toolbar.addAction(self.run_batch_action)

        toolbar.addSeparator()
        self.advanced_settings_action = QAction("Advanced Settings", self)
        self.advanced_settings_action.setCheckable(True)
        self.advanced_settings_action.toggled.connect(self._toggle_advanced_dock)
        toolbar.addAction(self.advanced_settings_action)

        self.original_panel = PreviewPanel("Original")
        self.cleaned_panel = PreviewPanel("Cleaned AI")
        self.outline_panel = PreviewPanel("Outline")
        self.reduced_panel = PreviewPanel("Reduced Colors")
        self.vector_panel = PreviewPanel("Vector SVG")

        self.preview_tabs = QTabWidget()
        self.preview_tabs.addTab(self.original_panel, "Original")
        self.preview_tabs.addTab(self.cleaned_panel, "Cleaned")
        self.preview_tabs.addTab(self.outline_panel, "Outline")
        self.preview_tabs.addTab(self.reduced_panel, "Reduced")
        self.preview_tabs.addTab(self.vector_panel, "Vector")

        preview_container = QWidget()
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(10)

        title_label = QLabel(APP_NAME)
        title_label.setStyleSheet("font-size: 22px; font-weight: 700; color: #111827;")
        subtitle_label = QLabel("Import -> Clean AI -> Detect Outline -> Reduce Colors -> Generate Vector -> Export SVG")
        subtitle_label.setStyleSheet("color: #4b5563;")
        preview_layout.addWidget(title_label)
        preview_layout.addWidget(subtitle_label)
        preview_layout.addWidget(self.preview_tabs, stretch=1)
        preview_container.setLayout(preview_layout)

        self.import_button = QPushButton("Import Image")
        self.import_button.clicked.connect(self.import_image)
        self.clean_button = QPushButton("Clean AI")
        self.clean_button.clicked.connect(self.clean_ai)
        self.outline_button = QPushButton("Detect Outline")
        self.outline_button.clicked.connect(self.detect_outline_stage)
        self.reduce_button = QPushButton("Reduce Colors")
        self.reduce_button.clicked.connect(self.reduce_colors_stage)
        self.vector_button = QPushButton("Generate Vector")
        self.vector_button.clicked.connect(self.generate_vector_stage)
        self.export_button = QPushButton("Export SVG")
        self.export_button.clicked.connect(self.export_svg)

        action_group = QGroupBox("Workflow")
        action_layout = QVBoxLayout()
        for button in [
            self.import_button,
            self.clean_button,
            self.outline_button,
            self.reduce_button,
            self.vector_button,
            self.export_button,
        ]:
            button.setMinimumHeight(36)
            action_layout.addWidget(button)
        action_group.setLayout(action_layout)

        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()
        self.a1111_url_edit = QLineEdit()
        self.enable_controlnet_check = QCheckBox("Use ControlNet Lineart")
        self.enable_controlnet_check.setChecked(True)
        self.preprocessor_combo = QComboBox()
        self.preprocessor_combo.addItem("Lineart", "lineart")
        self.preprocessor_combo.addItem("Canny", "canny")
        self.color_spin = QSpinBox()
        self.color_spin.setRange(2, 12)
        self.color_spin.setValue(6)
        self.output_dir_label = QLabel(str(self.runtime_config.output_dir(self.project_root)))
        self.output_dir_label.setWordWrap(True)
        settings_layout.addRow("A1111 URL", self.a1111_url_edit)
        settings_layout.addRow(self.enable_controlnet_check)
        settings_layout.addRow("Preprocessor", self.preprocessor_combo)
        settings_layout.addRow("Colors", self.color_spin)
        settings_layout.addRow("Output Folder", self.output_dir_label)
        settings_group.setLayout(settings_layout)

        info_group = QGroupBox("Status")
        info_layout = QVBoxLayout()
        self.runtime_hint = QLabel()
        self.runtime_hint.setWordWrap(True)
        self.runtime_hint.setStyleSheet("color: #4b5563;")
        self.stage_summary = QLabel("No image loaded.")
        self.stage_summary.setWordWrap(True)
        info_layout.addWidget(self.runtime_hint)
        info_layout.addWidget(self.stage_summary)
        info_group.setLayout(info_layout)

        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.addWidget(action_group)
        right_layout.addWidget(settings_group)
        right_layout.addWidget(info_group)
        right_layout.addStretch(1)
        right_panel.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(preview_container)
        splitter.addWidget(right_panel)
        splitter.setSizes([980, 320])

        self.console_log = QPlainTextEdit()
        self.console_log.setReadOnly(True)
        self.console_log.setMaximumBlockCount(500)
        self.console_log.setPlaceholderText("Process log will appear here.")

        console_group = QGroupBox("Console Log")
        console_layout = QVBoxLayout()
        console_layout.addWidget(self.console_log)
        console_group.setLayout(console_layout)

        central = QWidget()
        central_layout = QVBoxLayout()
        central_layout.addWidget(splitter, stretch=1)
        central_layout.addWidget(console_group, stretch=0)
        central.setLayout(central_layout)
        self.setCentralWidget(central)

        self.control_panel = TraceControlPanel()
        self.control_panel.sync_from_config(self.config)
        self.control_panel.settingsChanged.connect(self._on_advanced_controls_changed)
        self.control_panel.overlayChanged.connect(self._on_overlay_changed)

        self.advanced_dock = QDockWidget("Advanced Settings", self)
        self.advanced_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.advanced_dock.setWidget(self.control_panel)
        self.advanced_dock.visibilityChanged.connect(self._sync_advanced_action_state)
        self.addDockWidget(Qt.RightDockWidgetArea, self.advanced_dock)
        self.advanced_dock.hide()

        self.a1111_url_edit.textChanged.connect(self._on_basic_controls_changed)
        self.enable_controlnet_check.toggled.connect(self._on_basic_controls_changed)
        self.preprocessor_combo.currentIndexChanged.connect(self._on_basic_controls_changed)
        self.color_spin.valueChanged.connect(self._on_basic_controls_changed)

        self.statusBar().showMessage("Ready")

    def _sync_prompt_controls_from_config(self) -> None:
        self._updating_controls = True
        self.a1111_url_edit.blockSignals(True)
        self.enable_controlnet_check.blockSignals(True)
        self.preprocessor_combo.blockSignals(True)
        self.color_spin.blockSignals(True)
        self.a1111_url_edit.setText(self.config.controlnet.normalized_base_url())
        self.enable_controlnet_check.setChecked(bool(self.config.controlnet.enabled))
        self.color_spin.setValue(int(self.config.validated_color_count()))
        for index in range(self.preprocessor_combo.count()):
            if self.preprocessor_combo.itemData(index) == self.config.controlnet.validated_preprocessor():
                self.preprocessor_combo.setCurrentIndex(index)
                break
        self.a1111_url_edit.blockSignals(False)
        self.enable_controlnet_check.blockSignals(False)
        self.preprocessor_combo.blockSignals(False)
        self.color_spin.blockSignals(False)
        self._updating_controls = False

    def _apply_basic_controls_to_config(self) -> None:
        self.config.controlnet.base_url = str(self.a1111_url_edit.text()).strip()
        self.config.controlnet.enabled = bool(self.enable_controlnet_check.isChecked())
        self.config.controlnet.preprocessor = str(self.preprocessor_combo.currentData())
        self.config.color_count = int(self.color_spin.value())

    def _on_basic_controls_changed(self, *args) -> None:
        del args
        if self._updating_controls:
            return
        self._apply_basic_controls_to_config()
        self._updating_controls = True
        self.control_panel.sync_from_config(self.config)
        self._updating_controls = False
        self._invalidate_workflow(from_stage="cleanup")
        self._refresh_runtime_hint()

    def _on_advanced_controls_changed(self) -> None:
        if self._updating_controls:
            return
        self.control_panel.apply_to_config(self.config)
        self._sync_prompt_controls_from_config()
        self._invalidate_workflow(from_stage="cleanup")
        self._refresh_runtime_hint()

    def _toggle_advanced_dock(self, checked: bool) -> None:
        self.advanced_dock.setVisible(bool(checked))

    def _sync_advanced_action_state(self, visible: bool) -> None:
        self.advanced_settings_action.blockSignals(True)
        self.advanced_settings_action.setChecked(bool(visible))
        self.advanced_settings_action.blockSignals(False)

    def _refresh_runtime_hint(self) -> None:
        runtime = self.service.runtime_info()
        summary = [
            f"Detector: {runtime.detector_backend}",
            f"Real-ESRGAN: {runtime.realesrgan_backend}",
            f"U-Net: {runtime.unet_backend}",
            f"ControlNet: {'on' if self.config.controlnet.enabled else 'off'}",
        ]
        if runtime.warnings:
            summary.extend(runtime.warnings[:2])
        self.runtime_hint.setText(" | ".join(summary))

    def _refresh_button_states(self) -> None:
        has_input = self.current_image_path is not None
        self.clean_button.setEnabled(has_input)
        self.outline_button.setEnabled(has_input)
        self.reduce_button.setEnabled(has_input)
        self.vector_button.setEnabled(has_input)
        self.export_button.setEnabled(self.workflow.vector is not None)

    def _set_busy(self, busy: bool) -> None:
        for widget in [
            self.import_button,
            self.clean_button,
            self.outline_button,
            self.reduce_button,
            self.vector_button,
            self.export_button,
        ]:
            widget.setEnabled(not busy)
        for action in [
            self.auto_trace_action,
            self.export_cutline_action,
            self.export_vector_action,
            self.add_batch_action,
            self.run_batch_action,
            self.advanced_settings_action,
        ]:
            action.setEnabled(not busy)
        self.control_panel.setEnabled(not busy)
        self.a1111_url_edit.setEnabled(not busy)
        self.enable_controlnet_check.setEnabled(not busy)
        self.preprocessor_combo.setEnabled(not busy)
        self.color_spin.setEnabled(not busy)
        if not busy:
            self._refresh_button_states()

    def _log(self, message: str) -> None:
        self.console_log.appendPlainText(message)

    def import_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import T-Shirt Design Photo",
            str(self.project_root),
            "Image Files (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not file_path:
            return
        self._load_input_paths([Path(file_path)])

    def _load_input_paths(self, paths: list[Path]) -> None:
        valid_paths = [path for path in paths if path.suffix.lower() in IMAGE_SUFFIXES]
        if not valid_paths:
            QMessageBox.warning(self, "No Valid Images", "No supported image files were provided.")
            return

        self.current_image_path = valid_paths[0]
        self.last_result = None
        self.workflow = WorkflowState(input_path=self.current_image_path)

        try:
            loaded = load_image_data(self.current_image_path)
            resized, _ = resize_loaded_image(loaded, self.config.max_side_px)
            preview = compose_alpha_preview(resized.bgr, resized.alpha_mask)
        except Exception as error:
            QMessageBox.critical(self, "Image Error", str(error))
            return

        self.workflow.original_preview = preview
        self.original_panel.set_image(preview)
        self.cleaned_panel.clear()
        self.outline_panel.clear()
        self.reduced_panel.clear()
        self.vector_panel.clear()
        self.preview_tabs.setCurrentWidget(self.original_panel)
        self.stage_summary.setText(f"Loaded: {self.current_image_path.name}")
        self._log(f"Imported image: {self.current_image_path}")
        self._refresh_button_states()

    def clean_ai(self) -> None:
        if self.current_image_path is None:
            QMessageBox.information(self, "Import Required", "Import an image first.")
            return
        self._run_background_task("Running AI cleanup", self._build_cleanup_stage, self._on_cleanup_success)

    def detect_outline_stage(self) -> None:
        if self.current_image_path is None:
            QMessageBox.information(self, "Import Required", "Import an image first.")
            return
        self._run_background_task("Detecting outline", self._build_outline_stage, self._on_outline_success)

    def reduce_colors_stage(self) -> None:
        if self.current_image_path is None:
            QMessageBox.information(self, "Import Required", "Import an image first.")
            return
        self._run_background_task("Reducing colors", self._build_reduce_stage, self._on_reduce_success)

    def generate_vector_stage(self) -> None:
        if self.current_image_path is None:
            QMessageBox.information(self, "Import Required", "Import an image first.")
            return
        self._run_background_task("Generating vector", self._build_vector_stage, self._on_vector_success)

    def _run_background_task(self, description: str, callback, success_handler) -> None:
        self._set_busy(True)
        worker = CallableWorker(description, callback)
        worker.signals.started.connect(self._on_worker_started)
        worker.signals.success.connect(success_handler)
        worker.signals.failed.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _on_worker_started(self, description: str) -> None:
        self.statusBar().showMessage(description)
        self._log(description)

    def _on_worker_error(self, error_trace: str) -> None:
        self._set_busy(False)
        self.statusBar().showMessage("Process failed")
        self._log(error_trace.strip())
        QMessageBox.critical(self, "Processing Error", error_trace)

    def _build_cleanup_stage(self) -> dict[str, object]:
        preprocess = self.workflow.preprocess
        if preprocess is None:
            preprocess = prepare_photo_input(self.current_image_path, self.config, self.service.detector)

        cleanup = cleanup_image(
            preprocess.perspective_image,
            base_url=self.config.controlnet.normalized_base_url(),
            prompt=self.runtime_config.cleanup.prompt,
            negative_prompt=self.runtime_config.cleanup.negative_prompt,
            denoising_strength=self.runtime_config.cleanup.denoising_strength,
            steps=self.runtime_config.cleanup.steps,
            cfg_scale=self.runtime_config.cleanup.cfg_scale,
            connect_timeout_s=self.runtime_config.cleanup.connect_timeout_s,
            request_timeout_s=self.runtime_config.cleanup.request_timeout_s,
        )
        return {"preprocess": preprocess, "cleanup": cleanup}

    def _build_outline_stage(self) -> dict[str, object]:
        cleanup_bundle = self._build_cleanup_stage() if self.workflow.cleanup is None else {
            "preprocess": self.workflow.preprocess,
            "cleanup": self.workflow.cleanup,
        }
        cleanup = cleanup_bundle["cleanup"]
        outline = detect_outline(cleanup.cleaned_image, self.config)
        return {**cleanup_bundle, "outline": outline}

    def _build_reduce_stage(self) -> dict[str, object]:
        outline_bundle = self._build_outline_stage() if self.workflow.outline is None else {
            "preprocess": self.workflow.preprocess,
            "cleanup": self.workflow.cleanup,
            "outline": self.workflow.outline,
        }
        cleanup = outline_bundle["cleanup"]
        outline = outline_bundle["outline"]
        colors = reduce_palette(
            cleanup.cleaned_image,
            foreground_mask=outline.foreground_mask,
            color_count=self.config.validated_color_count(),
            config=self.config,
        )
        return {**outline_bundle, "colors": colors}

    def _build_vector_stage(self) -> dict[str, object]:
        reduce_bundle = self._build_reduce_stage() if self.workflow.colors is None else {
            "preprocess": self.workflow.preprocess,
            "cleanup": self.workflow.cleanup,
            "outline": self.workflow.outline,
            "colors": self.workflow.colors,
        }
        colors = reduce_bundle["colors"]
        vector = generate_vector_svg(
            label_map=colors.label_map,
            palette=colors.palette,
            canvas_size=(colors.quantized.shape[1], colors.quantized.shape[0]),
            config=self.config,
            title=f"{APP_NAME} - {self.current_image_path.stem}",
        )
        return {**reduce_bundle, "vector": vector}

    def _on_cleanup_success(self, payload: dict[str, object]) -> None:
        self._set_busy(False)
        self.workflow.preprocess = payload["preprocess"]
        self.workflow.cleanup = payload["cleanup"]
        self.workflow.outline = None
        self.workflow.colors = None
        self.workflow.vector = None
        self.workflow.exported_svg = None

        preprocess = self.workflow.preprocess
        cleanup = self.workflow.cleanup
        assert preprocess is not None
        assert cleanup is not None
        self.workflow.original_preview = preprocess.original_preview
        self.original_panel.set_image(preprocess.original_preview)
        self.cleaned_panel.set_image(cleanup.cleaned_image)
        self.outline_panel.clear()
        self.reduced_panel.clear()
        self.vector_panel.clear()
        self.preview_tabs.setCurrentWidget(self.cleaned_panel)

        detection = preprocess.metadata["detection"]
        perspective = preprocess.metadata["perspective"]
        self.stage_summary.setText(
            f"Cleaned AI ready | Detect: {detection['method']} | Perspective: {'yes' if perspective.get('applied') else 'no'}"
        )
        meta = cleanup.metadata
        if meta.get("usedFallback"):
            self._log(f"Clean AI fallback: {meta.get('fallbackReason', 'unknown reason')}")
        else:
            self._log(
                "Clean AI via Stable Diffusion WebUI "
                f"({meta.get('steps')} steps, CFG {meta.get('cfgScale')}, denoise {meta.get('denoisingStrength')})"
            )
        self.statusBar().showMessage("AI cleanup complete")

    def _on_outline_success(self, payload: dict[str, object]) -> None:
        self._on_cleanup_success({key: payload[key] for key in ("preprocess", "cleanup")})
        self.workflow.outline = payload["outline"]
        outline = self.workflow.outline
        assert outline is not None
        self.outline_panel.set_image(outline.preview_image)
        self.preview_tabs.setCurrentWidget(self.outline_panel)
        backend = outline.metadata.get("backend", "opencv_canny")
        self.stage_summary.setText(f"Outline ready | Backend: {backend}")
        if outline.metadata.get("fallbackReason"):
            self._log(f"Outline fallback: {outline.metadata['fallbackReason']}")
        else:
            self._log(f"Outline detected with {backend}")
        self.statusBar().showMessage("Outline detection complete")

    def _on_reduce_success(self, payload: dict[str, object]) -> None:
        self._on_outline_success({key: payload[key] for key in ("preprocess", "cleanup", "outline")})
        self.workflow.colors = payload["colors"]
        colors = self.workflow.colors
        assert colors is not None
        self.reduced_panel.set_image(colors.quantized)
        self.preview_tabs.setCurrentWidget(self.reduced_panel)
        self.stage_summary.setText(
            f"Reduced colors ready | Requested: {self.config.validated_color_count()} | Actual: {colors.metadata['actual_color_count']}"
        )
        self._log(
            "Color reduction complete "
            f"(requested {colors.metadata['requested_color_count']}, actual {colors.metadata['actual_color_count']})"
        )
        self.statusBar().showMessage("Color reduction complete")

    def _on_vector_success(self, payload: dict[str, object]) -> None:
        self._on_reduce_success({key: payload[key] for key in ("preprocess", "cleanup", "outline", "colors")})
        self.workflow.vector = payload["vector"]
        vector = self.workflow.vector
        assert vector is not None
        self.vector_panel.set_svg(vector.vector_svg)
        self.preview_tabs.setCurrentWidget(self.vector_panel)
        self.stage_summary.setText(f"Vector ready | Layers: {len(vector.layers)}")
        self._log(f"Vector SVG generated with {len(vector.layers)} printable layers.")
        self.statusBar().showMessage("Vector generation complete")
        self._refresh_button_states()

    def export_svg(self) -> None:
        if self.current_image_path is None:
            QMessageBox.information(self, "Import Required", "Import an image first.")
            return
        if self.workflow.vector is None:
            self._run_background_task("Generating vector for export", self._build_vector_stage, self._export_generated_vector)
            return
        self._export_current_vector()

    def _export_generated_vector(self, payload: dict[str, object]) -> None:
        self._on_vector_success(payload)
        self._export_current_vector()

    def _export_current_vector(self) -> None:
        assert self.workflow.vector is not None
        output_dir = self.runtime_config.output_dir(self.project_root)
        saved = export_timestamped_svg(self.workflow.vector.vector_svg, output_dir)
        self.workflow.exported_svg = saved
        self._log(f"SVG exported to {saved}")
        self.statusBar().showMessage(f"SVG exported: {saved}")
        self.stage_summary.setText(f"Export complete | {saved.name}")

    def run_auto_trace(self) -> None:
        if self.current_image_path is None:
            QMessageBox.information(self, "Import Required", "Import an image first.")
            return
        self._run_background_task("Running advanced auto trace", lambda: self.service.run(self.current_image_path), self._on_trace_success)

    def _on_trace_success(self, result: ProcessResult) -> None:
        self._set_busy(False)
        self.last_result = result
        self._render_process_result(result)
        total_ms = sum(result.timings_ms.values())
        self._log(f"Advanced auto trace complete in {total_ms:.0f} ms")
        self.statusBar().showMessage(f"Advanced auto trace complete in {total_ms:.0f} ms")

    def _render_process_result(self, result: ProcessResult) -> None:
        self.original_panel.set_image(result.previews["original"])
        self.cleaned_panel.set_image(result.previews.get("repaired", result.previews["original"]))
        outline_preview = result.previews.get("controlnet_outline", result.previews["edge"])
        if outline_preview.ndim == 2:
            outline_preview = cv2.cvtColor(outline_preview, cv2.COLOR_GRAY2BGR)
        self.outline_panel.set_image(outline_preview)
        self.reduced_panel.set_image(result.previews["quantized"])
        self.vector_panel.set_svg(result.vector_svg)
        self.preview_tabs.setCurrentWidget(self.vector_panel)
        self.stage_summary.setText(
            f"Advanced Auto Trace | Preset: {result.metadata.get('presetUsed', 'auto')} | Pipeline: {result.metadata.get('pipelineMode', 'photo')}"
        )

    def export_eps_pdf(self) -> None:
        if self.last_result is None:
            QMessageBox.information(self, "No Advanced Result", "Run Advanced Auto Trace before exporting EPS/PDF.")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Export EPS/PDF Directory", str(self.project_root))
        if not output_dir:
            return

        output_root = Path(output_dir)
        stem = self.last_result.input_path.stem
        svg_path = export_svg_file(output_root / f"{stem}.svg", self.last_result.vector_svg)
        eps_path = output_root / f"{stem}.eps"
        pdf_path = output_root / f"{stem}.pdf"
        try:
            conversion = convert_svg_to_eps_pdf(
                svg_path=svg_path,
                eps_path=eps_path,
                pdf_path=pdf_path,
                inkscape_bin=self.service.inkscape_bin,
            )
            exported = ", ".join(str(path) for path in conversion.values() if path is not None)
            self._log(f"EPS/PDF export complete: {exported}")
            self.statusBar().showMessage(f"Export complete: {exported}")
        except Exception as error:
            self._log(f"EPS/PDF export failed: {error}")
            QMessageBox.warning(self, "EPS/PDF Export", str(error))

    def export_cutline_svg(self) -> None:
        if self.last_result is None:
            QMessageBox.information(self, "No Advanced Result", "Run Advanced Auto Trace before exporting cutline.")
            return
        suggested = f"{self.last_result.input_path.stem}.cutline.svg"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Cutline SVG",
            str(self.project_root / suggested),
            "SVG Files (*.svg)",
        )
        if not output_path:
            return
        try:
            cutline_paths, _cutline_mask, cutline_meta = build_cutline_paths(
                self.last_result.layers,
                offset_px=self.config.settings.clamped().cutline_offset,
            )
            svg_text = compose_cutline_svg_string(
                canvas_size=(
                    self.last_result.previews["quantized"].shape[1],
                    self.last_result.previews["quantized"].shape[0],
                ),
                cutline_paths=cutline_paths,
                title=f"{APP_NAME} Cutline - {self.last_result.input_path.stem}",
            )
            saved = export_svg_file(output_path, svg_text)
            self._log(f"Cutline exported: {saved}")
            self.statusBar().showMessage(
                f"Cutline exported: {saved} | Paths: {cutline_meta['pathCount']} | Offset: {cutline_meta['offsetPx']} px"
            )
        except Exception as error:
            self._log(f"Cutline export failed: {error}")
            QMessageBox.warning(self, "Cutline Export", str(error))

    def add_batch_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Batch Images",
            str(self.project_root),
            "Image Files (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not files:
            return
        append_items = [Path(item) for item in files if Path(item).suffix.lower() in IMAGE_SUFFIXES]
        if not append_items:
            return
        self.batch_files.extend(append_items)
        self._log(f"Batch queue: {len(self.batch_files)} file(s)")
        self.statusBar().showMessage(f"Batch queue: {len(self.batch_files)} file(s)")

    def run_batch(self) -> None:
        if not self.batch_files:
            QMessageBox.information(self, "Batch Empty", "Add batch files first.")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Choose Output Folder", str(self.project_root))
        if not output_dir:
            return

        self._run_background_task(
            "Running advanced batch vectorization",
            lambda: self.batch_processor.run(
                input_paths=self.batch_files,
                output_dir=Path(output_dir),
                export_formats=("svg", "eps", "pdf"),
            ),
            self._on_batch_success,
        )

    def _on_batch_success(self, results: list[ProcessResult]) -> None:
        self._set_busy(False)
        if not results:
            self._log("Batch complete with no files.")
            self.statusBar().showMessage("Batch complete with no files.")
            return
        self.last_result = results[-1]
        self._render_process_result(self.last_result)
        self._log(f"Batch complete: {len(results)} file(s) processed.")
        self.statusBar().showMessage(f"Batch complete: {len(results)} file(s) processed.")

    def _invalidate_workflow(self, from_stage: str) -> None:
        if self.current_image_path is None:
            return
        if from_stage == "cleanup":
            self.workflow.preprocess = None
            self.workflow.cleanup = None
            self.workflow.outline = None
            self.workflow.colors = None
            self.workflow.vector = None
            self.workflow.exported_svg = None
            self.cleaned_panel.clear()
            self.outline_panel.clear()
            self.reduced_panel.clear()
            self.vector_panel.clear()
            self.stage_summary.setText(f"Settings changed. Re-run Clean AI for {self.current_image_path.name}.")
            self._log("Settings changed. Downstream prompt workflow stages were invalidated.")
            self._refresh_button_states()

    def _on_overlay_changed(self, enabled: bool) -> None:
        del enabled
        if self.last_result is not None:
            self._render_process_result(self.last_result)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths: list[Path] = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path)
        if paths:
            self._load_input_paths(paths)
            event.acceptProposedAction()
