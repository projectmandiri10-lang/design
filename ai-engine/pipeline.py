from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from vectorizer.cleanup import cleanup_image
from vectorizer.color_reduce import reduce_colors
from vectorizer.io_utils import copy_file, ensure_dir, read_image, save_json, write_image
from vectorizer.perspective_fix import correct_perspective
from vectorizer.smooth_svg import smooth_svg
from vectorizer.texture_remove import remove_texture
from vectorizer.upscale import Upscaler
from vectorizer.vectorize import check_potrace, vectorize_bitmap


def emit(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


class PipelineRunner:
    def __init__(self, models_dir: str | Path, potrace_bin: str):
        self.models_dir = Path(models_dir)
        self.potrace_bin = potrace_bin
        self.upscaler = Upscaler(self.models_dir)

    def health(self):
        potrace_health = check_potrace(self.potrace_bin)
        upscale_health = self.upscaler.health()

        warnings = []
        if not potrace_health.get("available"):
            warnings.append("Potrace is not available. Vectorization will fail until it is installed.")
        if not upscale_health.get("modelExists"):
            warnings.append("RealESRGAN_x4plus.pth not found. The pipeline will use classical CPU upscaling.")
        if not upscale_health.get("realesrganAvailable"):
            warnings.append("Real-ESRGAN Python dependencies are not available. Classical CPU upscaling will be used.")

        return {
            "python": sys.version.split()[0],
            "potrace": potrace_health,
            "upscale": upscale_health,
        }, warnings

    def run_job(self, job_id: str, input_path: str | Path, color_count: int, output_dir: str | Path):
        output_root = ensure_dir(output_dir)
        input_path = Path(input_path)

        timings = {}
        warnings = []
        fallbacks = []

        health, startup_warnings = self.health()
        warnings.extend(startup_warnings)

        original_name = f"original{input_path.suffix.lower() or '.png'}"
        original_copy = copy_file(input_path, output_root / original_name)

        image = read_image(input_path)

        def run_stage(stage_name, stage_callable):
            started = time.perf_counter()
            result = stage_callable()
            timings[stage_name] = round((time.perf_counter() - started) * 1000, 2)
            return result

        perspective_image, perspective_meta = run_stage("perspectiveFix", lambda: correct_perspective(image))
        texture_image, texture_meta = run_stage("textureRemove", lambda: remove_texture(perspective_image))
        upscaled_image, upscale_meta = run_stage("upscale", lambda: self.upscaler.upscale(texture_image))
        cleaned_image, design_mask, cleanup_meta = run_stage("cleanup", lambda: cleanup_image(upscaled_image))
        quantized_image, label_map, palette, color_meta = run_stage(
            "colorReduce",
            lambda: reduce_colors(cleaned_image, design_mask, color_count),
        )

        processed_preview_name = "processed-preview.png"
        run_stage("savePreview", lambda: write_image(output_root / processed_preview_name, quantized_image))

        raw_svg_name = "raw-result.svg"
        vector_meta = run_stage(
            "vectorize",
            lambda: vectorize_bitmap(
                quantized_image,
                label_map,
                palette,
                output_root / raw_svg_name,
                potrace_bin=self.potrace_bin,
            ),
        )

        final_svg_name = "result.svg"
        smooth_meta = run_stage("smoothSvg", lambda: smooth_svg(output_root / raw_svg_name, output_root / final_svg_name))

        if upscale_meta.get("fallbackUsed"):
            fallbacks.append("classical_upscale")
            if upscale_meta.get("fallbackReason"):
                warnings.append(upscale_meta["fallbackReason"])

        if smooth_meta.get("fallbackUsed"):
            fallbacks.append("raw_svg_passthrough")
            if smooth_meta.get("fallbackReason"):
                warnings.append(smooth_meta["fallbackReason"])

        svg_content = (output_root / final_svg_name).read_text(encoding="utf-8")

        manifest = {
            "jobId": job_id,
            "artifacts": {
                "original": original_copy.name,
                "processedPreview": processed_preview_name,
                "rawSvg": raw_svg_name,
                "svg": final_svg_name,
                "manifest": "result.json",
            },
            "palette": palette,
            "timings": timings,
            "warnings": sorted(set(warnings)),
            "fallbacks": sorted(set(fallbacks)),
            "metadata": {
                "health": health,
                "perspective": perspective_meta,
                "texture": texture_meta,
                "upscale": upscale_meta,
                "cleanup": cleanup_meta,
                "colorReduce": color_meta,
                "vectorize": vector_meta,
                "smoothSvg": smooth_meta,
                "inputShape": [int(image.shape[1]), int(image.shape[0])],
                "outputShape": [int(quantized_image.shape[1]), int(quantized_image.shape[0])],
            },
            "svgContent": svg_content,
        }

        save_json(output_root / "result.json", manifest)
        return manifest


def run_worker(models_dir: str | Path, potrace_bin: str):
    runner = PipelineRunner(models_dir=models_dir, potrace_bin=potrace_bin)
    health, warnings = runner.health()
    emit({"type": "ready", "health": health, "warnings": warnings})

    for line in sys.stdin:
        if not line.strip():
            continue

        request_id = None
        try:
            payload = json.loads(line)
            request_id = payload.get("requestId")
            result = runner.run_job(
                job_id=payload["jobId"],
                input_path=payload["inputPath"],
                color_count=int(payload["colorCount"]),
                output_dir=payload["outputDir"],
            )
            emit({"type": "result", "requestId": request_id, "success": True, "payload": result})
        except Exception as error:  # pragma: no cover - worker boundary
            emit(
                {
                    "type": "result",
                    "requestId": request_id,
                    "success": False,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )


def run_once(args):
    runner = PipelineRunner(models_dir=args.models_dir, potrace_bin=args.potrace_bin)
    result = runner.run_job(
        job_id=args.job_id or "manual-run",
        input_path=args.input,
        color_count=args.color_count,
        output_dir=args.output_dir,
    )
    sys.stdout.write(json.dumps(result, indent=2))
    sys.stdout.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="AI screen printing vectorizer pipeline")
    parser.add_argument("--worker", action="store_true", help="Run in persistent worker mode.")
    parser.add_argument("--input", help="Input image path for one-shot mode.")
    parser.add_argument("--output-dir", help="Output directory for one-shot mode.")
    parser.add_argument("--color-count", type=int, default=4, help="Color count for one-shot mode.")
    parser.add_argument("--job-id", help="Optional job id for one-shot mode.")
    parser.add_argument("--models-dir", default=str(Path(__file__).resolve().parents[1] / "models"))
    parser.add_argument("--potrace-bin", default="potrace")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        run_worker(args.models_dir, args.potrace_bin)
        return

    if not args.input or not args.output_dir:
        raise SystemExit("--input and --output-dir are required outside worker mode.")

    run_once(args)


if __name__ == "__main__":
    main()
