from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from export.export_eps import convert_svg_to_eps_pdf, detect_inkscape_binary
from export.export_svg import compose_svg_string, export_svg_file

from .color_reduction import reduce_colors
from .controlnet_client import ControlNetClient, ControlNetClientError
from .detect_sablon_area import SablonAreaDetector
from .fabric_texture_removal import FabricDenoiserUNet
from .image_preprocess import (
    RealESRGANRestorer,
    boost_trace_inputs,
    compose_alpha_preview,
    compose_segment_preview,
    load_image_data,
    preprocess_ai_outline,
    preprocess_artwork,
    preprocess_for_outline,
    preprocess_raster_artwork,
    resize_loaded_image,
)
from .perspective_correction import correct_perspective
from .repair_cracks import repair_cracks
from .trace_analysis import analyze_trace_input, resolve_preset, select_pipeline_mode
from .types import (
    DetectionResult,
    LoadedImage,
    ModelRuntimeInfo,
    PipelineConfig,
    ProcessResult,
    TraceAnalysis,
)
from .vectorize_bitmap import detect_potrace_binary, vectorize_by_color_layers


class AutoTraceService:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.detector = SablonAreaDetector(config.paths.detector_model)
        self.restorer = RealESRGANRestorer(config.paths.realesrgan_model)
        self.denoiser = FabricDenoiserUNet(config.paths.unet_model)
        self.potrace_bin = detect_potrace_binary(config.paths.potrace_bin)
        self.inkscape_bin = detect_inkscape_binary(config.paths.inkscape_bin)
        self._cache: dict[str, dict[tuple[object, ...], object]] = {
            "loaded": {},
            "resized": {},
            "analysis": {},
            "detection": {},
            "preprocess": {},
            "color": {},
        }

    def runtime_info(self) -> ModelRuntimeInfo:
        warnings: list[str] = []
        for message in [self.detector.runtime_warning, self.restorer.runtime_warning, self.denoiser.runtime_warning]:
            if message:
                warnings.append(message)
        if self.potrace_bin is None:
            warnings.append("Potrace binary not found. Vectorization falls back to contour tracing.")
        if self.inkscape_bin is None:
            warnings.append("Inkscape not found. EPS/PDF export is unavailable until configured.")

        return ModelRuntimeInfo(
            detector_loaded=self.detector.loaded,
            realesrgan_loaded=self.restorer.loaded,
            unet_loaded=self.denoiser.loaded,
            detector_backend=self.detector.backend,
            realesrgan_backend=self.restorer.backend,
            unet_backend=self.denoiser.backend,
            warnings=warnings,
        )

    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path | None = None,
        export_formats: tuple[str, ...] | list[str] | None = None,
        export_basename: str | None = None,
    ) -> ProcessResult:
        input_file = Path(input_path)
        timings: dict[str, float] = {}
        warnings: list[str] = []
        cache_hits: dict[str, bool] = {}
        runtime = self.runtime_info()
        warnings.extend(runtime.warnings)
        boost_meta: dict[str, object] = {
            "applied": False,
            "scale": 1.0,
            "sourceMaxSide": 0,
            "targetMaxSide": 0,
        }
        controlnet_meta: dict[str, object] = {
            "requested": bool(self.config.controlnet.enabled),
            "applied": False,
            "preprocessor": self.config.controlnet.validated_preprocessor(),
            "model": self.config.controlnet.resolved_model_prefix(),
            "baseUrl": self.config.controlnet.normalized_base_url(),
            "fallbackReason": None,
        }
        controlnet_outline: np.ndarray | None = None
        outline_backend = "classic"

        def stage(name: str, func: Callable[[], Any]):
            started = time.perf_counter()
            result = func()
            timings[name] = round((time.perf_counter() - started) * 1000, 2)
            return result

        image_state = self._image_state_key(input_file)
        loaded, cache_hits["loaded"] = stage("load_image", lambda: self._get_loaded(input_file, image_state))
        resized_loaded, resize_scale, cache_hits["resized"] = stage(
            "resize",
            lambda: self._get_resized(loaded, image_state),
        )
        analysis, raster_artwork, cache_hits["analysis"] = stage(
            "analysis",
            lambda: self._get_analysis(resized_loaded, image_state),
        )
        preset_used = resolve_preset(self.config.validated_preset(), analysis)
        pipeline_mode = select_pipeline_mode(preset_used, resized_loaded, analysis)
        skipped_stages: list[str] = []

        detection, cache_hits["detection"] = stage(
            "detect_sablon_area",
            lambda: self._get_detection(
                resized_loaded=resized_loaded,
                image_state=image_state,
                pipeline_mode=pipeline_mode,
                preset_used=preset_used,
                raster_foreground=raster_artwork.foreground_mask,
            ),
        )
        detection, boost_meta = stage(
            "small_image_boost",
            lambda: self._boost_detection_inputs(detection),
        )

        if pipeline_mode == "artwork":
            perspective = detection.cropped_image
            perspective_meta = {"applied": False, "skipped": True, "reason": "artwork_mode"}
            restored = detection.cropped_image
            restore_meta = {"backend": "skipped", "fallback": False, "reason": "artwork_mode"}
            denoised = detection.cropped_image
            denoise_meta = {"backend": "skipped", "fallback": False, "reason": "artwork_mode"}
            repaired = detection.cropped_image
            repair_meta = {"crack_fill_ratio": 0.0, "skipped": True, "reason": "artwork_mode"}
            preprocess_final, cache_hits["preprocess"] = stage(
                "bitmap_preparation",
                lambda: self._get_preprocess(
                    pipeline_mode=pipeline_mode,
                    image=detection.cropped_image,
                    mask=detection.mask,
                    image_state=image_state,
                    preset_used=preset_used,
                ),
            )
            skipped_stages.extend(
                [
                    "controlnet_outline",
                    "perspective_correction",
                    "realesrgan_restore",
                    "unet_texture_remove",
                    "repair_cracks",
                ]
            )
            if controlnet_meta["requested"]:
                controlnet_meta["fallbackReason"] = "ControlNet is only applied to the photo pipeline."
            min_component_ratio = self.config.min_component_ratio_artwork
        elif pipeline_mode == "raster_artwork":
            perspective = detection.cropped_image
            perspective_meta = {"applied": False, "skipped": True, "reason": "raster_artwork_mode"}
            restored = detection.cropped_image
            restore_meta = {"backend": "skipped", "fallback": False, "reason": "raster_artwork_mode"}
            denoised = detection.cropped_image
            denoise_meta = {"backend": "skipped", "fallback": False, "reason": "raster_artwork_mode"}
            repaired = detection.cropped_image
            repair_meta = {"crack_fill_ratio": 0.0, "skipped": True, "reason": "raster_artwork_mode"}
            preprocess_final, cache_hits["preprocess"] = stage(
                "bitmap_preparation",
                lambda: self._get_preprocess(
                    pipeline_mode=pipeline_mode,
                    image=detection.cropped_image,
                    mask=detection.mask,
                    image_state=image_state,
                    preset_used=preset_used,
                ),
            )
            skipped_stages.extend(
                [
                    "controlnet_outline",
                    "perspective_correction",
                    "realesrgan_restore",
                    "unet_texture_remove",
                    "repair_cracks",
                ]
            )
            if controlnet_meta["requested"]:
                controlnet_meta["fallbackReason"] = "ControlNet is only applied to the photo pipeline."
            min_component_ratio = self.config.min_component_ratio_raster_artwork
        else:
            perspective, perspective_meta = stage(
                "perspective_correction",
                lambda: correct_perspective(detection.cropped_image),
            )
            prepared_photo, cache_hits["preprocess"] = stage(
                "bitmap_preparation",
                lambda: self._get_preprocess(
                    pipeline_mode="photo_initial",
                    image=perspective,
                    mask=detection.mask,
                    image_state=image_state,
                    preset_used=preset_used,
                ),
            )
            restored, restore_meta = stage("realesrgan_restore", lambda: self.restorer.restore(perspective))
            denoised, denoise_meta = stage("unet_texture_remove", lambda: self.denoiser.remove(restored))
            repaired, repair_meta = stage("repair_cracks", lambda: repair_cracks(denoised))
            if self.config.controlnet.enabled:
                try:
                    controlnet_outline, controlnet_meta = stage(
                        "controlnet_outline",
                        lambda: self._run_controlnet_outline(repaired),
                    )
                    preprocess_final = stage(
                        "region_segmentation",
                        lambda: preprocess_ai_outline(controlnet_outline, self.config, preset_used=preset_used),
                    )
                    outline_backend = "controlnet"
                except ControlNetClientError as error:
                    warnings.append(f"ControlNet fallback: {error}")
                    controlnet_meta = {
                        **controlnet_meta,
                        "requested": True,
                        "applied": False,
                        "fallbackReason": str(error),
                    }
                    preprocess_final = stage(
                        "region_segmentation",
                        lambda: preprocess_for_outline(repaired, self.config, preset_used=preset_used),
                    )
                    preprocess_final = _merge_photo_preprocess(prepared_photo, preprocess_final)
            else:
                skipped_stages.append("controlnet_outline")
                preprocess_final = stage(
                    "region_segmentation",
                    lambda: preprocess_for_outline(repaired, self.config, preset_used=preset_used),
                )
                preprocess_final = _merge_photo_preprocess(prepared_photo, preprocess_final)
            min_component_ratio = self.config.min_component_ratio_photo

        min_region_area = self._resolved_min_shape_area(preset_used)
        quantized, label_map, palette, color_meta, cache_hits["color"] = stage(
            "reduce_colors",
            lambda: self._get_color_reduction(
                image=repaired,
                foreground_mask=preprocess_final.foreground_mask,
                image_state=image_state,
                preset_used=preset_used,
            ),
        )
        layers, vector_meta = stage(
            "vectorize",
            lambda: vectorize_by_color_layers(
                label_map=label_map,
                palette=palette,
                min_component_ratio=min_component_ratio,
                potrace_bin=self.potrace_bin,
                minimum_area_floor=min_region_area,
                settings=self.config.settings,
                quality_mode=self.config.validated_quality_mode(),
                preset=preset_used,
            ),
        )

        if not layers:
            raise RuntimeError("Vectorization produced no layers.")

        svg_content = stage(
            "compose_svg",
            lambda: compose_svg_string(
                canvas_size=(quantized.shape[1], quantized.shape[0]),
                layers=layers,
                title=f"AUTO VECTOR SABLON AI - {input_file.stem}",
                background_mode=self.config.validated_background_mode(),
            ),
        )

        exports: dict[str, Path] = {}
        if output_dir is not None:
            requested = tuple(format_name.lower() for format_name in (export_formats or ("svg",)))
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            stem = export_basename or input_file.stem

            if "svg" in requested or "eps" in requested or "pdf" in requested:
                svg_target = output_path / f"{stem}.svg"
                stage("export_svg", lambda: export_svg_file(svg_target, svg_content))
                exports["svg"] = svg_target

            if "eps" in requested or "pdf" in requested:
                eps_target = output_path / f"{stem}.eps" if "eps" in requested else None
                pdf_target = output_path / f"{stem}.pdf" if "pdf" in requested else None
                try:
                    conversion = stage(
                        "export_eps_pdf",
                        lambda: convert_svg_to_eps_pdf(
                            svg_path=exports["svg"],
                            eps_path=eps_target,
                            pdf_path=pdf_target,
                            inkscape_bin=self.inkscape_bin,
                        ),
                    )
                    if conversion.get("eps"):
                        exports["eps"] = Path(str(conversion["eps"]))
                    if conversion.get("pdf"):
                        exports["pdf"] = Path(str(conversion["pdf"]))
                except Exception as error:
                    warnings.append(f"EPS/PDF export skipped: {error}")

        original_preview = compose_alpha_preview(resized_loaded.bgr, resized_loaded.alpha_mask)
        segmented_preview = compose_segment_preview(quantized, preprocess_final.edge_map)
        previews = {
            "original": original_preview,
            "detected": detection.cropped_image,
            "edge": preprocess_final.edge_map,
            "segmented": segmented_preview,
            "quantized": quantized,
            "repaired": repaired,
        }
        if controlnet_outline is not None:
            previews["controlnet_outline"] = controlnet_outline
        metadata = {
            "pipelineMode": pipeline_mode,
            "presetUsed": preset_used,
            "qualityMode": self.config.validated_quality_mode(),
            "backgroundMode": self.config.validated_background_mode(),
            "outlineBackend": outline_backend,
            "controlnet": controlnet_meta,
            "analysis": analysis.to_metadata(),
            "alphaInfo": {
                "hasAlpha": resized_loaded.has_alpha,
                "transparentRatio": resized_loaded.transparent_ratio,
            },
            "rasterArtwork": {
                "whiteBackgroundRatio": raster_artwork.white_background_ratio,
                "cornerWhiteRatio": raster_artwork.corner_white_ratio,
                "foregroundRatio": raster_artwork.foreground_ratio,
                "meanSaturation": raster_artwork.mean_saturation,
                "whiteCutoff": raster_artwork.white_cutoff,
                "coarseColorBins": raster_artwork.coarse_color_bins,
                "edgeDensity": raster_artwork.edge_density,
                "maxSide": raster_artwork.max_side,
                "isCandidate": raster_artwork.is_candidate,
            },
            "smallImageBoost": boost_meta,
            "skippedStages": skipped_stages,
            "resize_scale": resize_scale,
            "cacheHits": cache_hits,
            "detection": {
                "bbox_xywh": detection.bbox_xywh,
                "confidence": detection.confidence,
                "method": detection.method,
            },
            "perspective": perspective_meta,
            "realesrgan": restore_meta,
            "unet": denoise_meta,
            "repair": repair_meta,
            "color": color_meta,
            "vectorize": vector_meta,
            "cleanupStats": vector_meta.get("cleanupStats", {}),
            "nodeReductionStats": vector_meta.get("nodeReductionStats", {}),
        }

        return ProcessResult(
            input_path=input_file,
            timings_ms=timings,
            warnings=sorted(set(warnings)),
            runtime_info=runtime,
            previews=previews,
            layers=layers,
            exports=exports,
            palette=palette,
            vector_svg=svg_content,
            metadata=metadata,
        )

    def _image_state_key(self, input_file: Path) -> tuple[object, ...]:
        resolved = str(input_file.resolve())
        stat = input_file.stat()
        return resolved, int(stat.st_mtime_ns), int(stat.st_size), int(self.config.max_side_px)

    def _cache_lookup(self, bucket: str, key: tuple[object, ...], factory: Callable[[], Any]) -> tuple[Any, bool]:
        cache_bucket = self._cache[bucket]
        if key in cache_bucket:
            return cache_bucket[key], True
        value = factory()
        cache_bucket[key] = value
        return value, False

    def _get_loaded(self, input_file: Path, image_state: tuple[object, ...]) -> tuple[LoadedImage, bool]:
        return self._cache_lookup("loaded", image_state, lambda: load_image_data(input_file))

    def _get_resized(
        self,
        loaded: LoadedImage,
        image_state: tuple[object, ...],
    ) -> tuple[LoadedImage, float, bool]:
        key = (*image_state, int(self.config.max_side_px))

        def builder() -> tuple[LoadedImage, float]:
            return resize_loaded_image(loaded, self.config.max_side_px)

        value, hit = self._cache_lookup("resized", key, builder)
        resized_loaded, resize_scale = value
        return resized_loaded, resize_scale, hit

    def _get_analysis(
        self,
        resized_loaded: LoadedImage,
        image_state: tuple[object, ...],
    ) -> tuple[TraceAnalysis, Any, bool]:
        key = (*image_state, "analysis", resized_loaded.bgr.shape[1], resized_loaded.bgr.shape[0])
        value, hit = self._cache_lookup("analysis", key, lambda: analyze_trace_input(resized_loaded, self.config))
        analysis, raster_artwork = value
        return analysis, raster_artwork, hit

    def _get_detection(
        self,
        resized_loaded: LoadedImage,
        image_state: tuple[object, ...],
        pipeline_mode: str,
        preset_used: str,
        raster_foreground: np.ndarray,
    ) -> tuple[DetectionResult, bool]:
        key = (*image_state, "detect", pipeline_mode, preset_used)

        def builder() -> DetectionResult:
            if pipeline_mode == "artwork":
                return self._detect_from_alpha(resized_loaded)
            if pipeline_mode == "raster_artwork":
                return self._detect_from_mask(
                    resized_loaded.bgr,
                    raster_foreground,
                    method="raster_foreground_bbox",
                )
            return self.detector.detect(resized_loaded.bgr)

        return self._cache_lookup("detection", key, builder)

    def _get_preprocess(
        self,
        pipeline_mode: str,
        image: np.ndarray,
        mask: np.ndarray,
        image_state: tuple[object, ...],
        preset_used: str,
    ) -> tuple[Any, bool]:
        key = (*image_state, "preprocess", pipeline_mode, preset_used, *self.config.settings_signature(), image.shape[1], image.shape[0])

        def builder():
            if pipeline_mode == "artwork":
                return preprocess_artwork(image, mask, self.config, preset_used=preset_used)
            if pipeline_mode == "raster_artwork":
                return preprocess_raster_artwork(image, mask, self.config, preset_used=preset_used)
            return preprocess_for_outline(image, self.config, preset_used=preset_used)

        return self._cache_lookup("preprocess", key, builder)

    def _get_color_reduction(
        self,
        image: np.ndarray,
        foreground_mask: np.ndarray,
        image_state: tuple[object, ...],
        preset_used: str,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], dict[str, object], bool]:
        key = (
            *image_state,
            "color",
            image.shape[1],
            image.shape[0],
            *self.config.settings_signature(),
            self._resolved_min_shape_area(preset_used),
            preset_used,
        )

        def builder():
            return reduce_colors(
                image=image,
                color_count=self.config.validated_color_count(),
                mask=foreground_mask,
                min_region_area=self._resolved_min_shape_area(preset_used),
                background_mode=self.config.validated_background_mode(),
                ignore_white=self.config.settings.clamped().ignore_white,
                quality_mode=self.config.validated_quality_mode(),
            )

        value, hit = self._cache_lookup("color", key, builder)
        quantized, label_map, palette, color_meta = value
        return quantized, label_map, palette, color_meta, hit

    def _resolved_min_shape_area(self, preset_used: str | None = None) -> int:
        settings = self.config.settings.clamped()
        base = settings.min_shape_area
        detail_bonus = max(0, 40 - settings.detail) // 4
        despeckle_bonus = settings.despeckle // 3
        quality_bonus = 4 if self.config.validated_quality_mode() == "high_quality" else 0
        resolved = max(1, int(base + detail_bonus + despeckle_bonus - quality_bonus))
        if preset_used == "text_typo":
            return max(1, int(round(resolved * 0.55)))
        if preset_used == "dtf_screen_print":
            return max(1, int(round(resolved * 0.8)))
        if preset_used == "sticker_cutting":
            return max(1, int(round(resolved * 1.35)))
        return resolved

    def _detect_from_alpha(self, loaded: LoadedImage) -> DetectionResult:
        assert loaded.alpha_mask is not None
        mask = np.where(loaded.alpha_mask > 0, 255, 0).astype(np.uint8)
        return self._detect_from_mask(loaded.bgr, mask, method="alpha_bbox")

    def _detect_from_mask(self, image: np.ndarray, mask: np.ndarray, method: str) -> DetectionResult:
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        points = cv2.findNonZero(mask)
        if points is None:
            full_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            return DetectionResult(
                cropped_image=image.copy(),
                bbox_xywh=(0, 0, image.shape[1], image.shape[0]),
                confidence=0.0,
                method=f"{method}_fallback_full",
                mask=full_mask,
            )

        x, y, w, h = cv2.boundingRect(points)
        pad_x = max(2, int(w * 0.02))
        pad_y = max(2, int(h * 0.02))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(image.shape[1], x + w + pad_x)
        y1 = min(image.shape[0], y + h + pad_y)

        roi_image = image[y0:y1, x0:x1].copy()
        roi_mask = mask[y0:y1, x0:x1].copy()
        coverage = float(np.count_nonzero(mask)) / float(mask.size)

        return DetectionResult(
            cropped_image=roi_image,
            bbox_xywh=(x0, y0, x1 - x0, y1 - y0),
            confidence=round(coverage, 4),
            method=method,
            mask=roi_mask,
        )

    def _boost_detection_inputs(self, detection: DetectionResult) -> tuple[DetectionResult, dict[str, object]]:
        if not self.config.settings.clamped().small_image_boost:
            return detection, {
                "applied": False,
                "scale": 1.0,
                "sourceMaxSide": max(detection.cropped_image.shape[:2]),
                "targetMaxSide": max(detection.cropped_image.shape[:2]),
            }

        boosted_image, boosted_mask, meta = boost_trace_inputs(detection.cropped_image, detection.mask, self.config)
        if not meta.get("applied"):
            return detection, meta

        return DetectionResult(
            cropped_image=boosted_image,
            bbox_xywh=detection.bbox_xywh,
            confidence=detection.confidence,
            method=detection.method,
            mask=boosted_mask if boosted_mask is not None else detection.mask,
        ), meta

    def _run_controlnet_outline(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        client = ControlNetClient(self.config.controlnet)
        result = client.render_outline(image)
        return result.image, {
            "requested": True,
            "applied": True,
            "preprocessor": result.discovery.module,
            "model": result.discovery.model,
            "baseUrl": result.discovery.base_url,
            "fallbackReason": None,
            "version": result.discovery.version,
            "requestSize": {
                "width": int(result.request_size[0]),
                "height": int(result.request_size[1]),
            },
        }

    def palette_preview(self, result: ProcessResult) -> np.ndarray:
        color_height = 50
        width = max(1, result.previews["quantized"].shape[1])
        canvas = np.full((color_height, width, 3), 255, dtype=np.uint8)
        if not result.palette:
            return canvas

        total = sum(int(item["pixels"]) for item in result.palette) or len(result.palette)
        x = 0
        for item in result.palette:
            ratio = int(item["pixels"]) / total
            block_w = max(1, int(width * ratio))
            rgb = item["rgb"]
            color = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.uint8)
            canvas[:, x : min(width, x + block_w)] = color
            x += block_w
            if x >= width:
                break
        return canvas


def _merge_photo_preprocess(primary, secondary):
    return type(primary)(
        gray=secondary.gray,
        enhanced_gray=secondary.enhanced_gray,
        edge_map=cv2.bitwise_or(primary.edge_map, secondary.edge_map),
        threshold_map=cv2.bitwise_or(primary.threshold_map, secondary.threshold_map),
        foreground_mask=cv2.bitwise_or(primary.foreground_mask, secondary.foreground_mask),
    )
