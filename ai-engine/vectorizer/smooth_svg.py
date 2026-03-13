from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from svgpathtools import Arc, CubicBezier, Line, Path as SVGPath, QuadraticBezier, parse_path
except ImportError:  # pragma: no cover - optional at runtime
    Arc = CubicBezier = Line = SVGPath = QuadraticBezier = parse_path = None

try:  # pragma: no cover - optional experimental runtime
    import pydiffvg as diffvg
except ImportError:  # pragma: no cover - optional runtime
    diffvg = None

SVG_NS = "http://www.w3.org/2000/svg"


def _is_collinear(first: complex, second: complex, third: complex, tolerance: float = 1.5) -> bool:
    vector_one = second - first
    vector_two = third - first
    if abs(vector_two) < 1e-6:
        return True
    cross = abs(vector_one.real * vector_two.imag - vector_one.imag * vector_two.real)
    return cross <= tolerance


def _clamp_handle(start: complex, control: complex, end: complex, ratio: float = 0.7) -> complex:
    chord = end - start
    chord_length = abs(chord)
    if chord_length < 1e-6:
        return control

    handle = control - start
    handle_length = abs(handle)
    max_length = chord_length * ratio
    if handle_length <= max_length:
        return control
    return start + handle / handle_length * max_length


def _estimate_bbox_area(path_obj: SVGPath) -> float:
    min_x, max_x, min_y, max_y = path_obj.bbox()
    return max(max_x - min_x, 0.0) * max(max_y - min_y, 0.0)


def _simplify_path(path_obj: SVGPath) -> SVGPath:
    simplified_segments = []

    for segment in path_obj:
        if segment.length(error=1e-3) < 1.0:
            continue

        if isinstance(segment, Line):
            if simplified_segments and isinstance(simplified_segments[-1], Line):
                previous = simplified_segments[-1]
                if _is_collinear(previous.start, previous.end, segment.end):
                    simplified_segments[-1] = Line(previous.start, segment.end)
                    continue
            simplified_segments.append(segment)
            continue

        if isinstance(segment, CubicBezier):
            if (
                _is_collinear(segment.start, segment.control1, segment.end)
                and _is_collinear(segment.start, segment.control2, segment.end)
            ):
                simplified_segments.append(Line(segment.start, segment.end))
                continue

            control1 = _clamp_handle(segment.start, segment.control1, segment.end)
            control2 = _clamp_handle(segment.end, segment.control2, segment.start)
            simplified_segments.append(CubicBezier(segment.start, control1, control2, segment.end))
            continue

        if isinstance(segment, QuadraticBezier):
            if _is_collinear(segment.start, segment.control, segment.end):
                simplified_segments.append(Line(segment.start, segment.end))
            else:
                simplified_segments.append(segment)
            continue

        if isinstance(segment, Arc):
            simplified_segments.append(segment)

    return SVGPath(*simplified_segments)


def smooth_svg(input_svg_path: str | Path, output_svg_path: str | Path):
    input_path = Path(input_svg_path)
    output_path = Path(output_svg_path)

    if parse_path is None:
        shutil.copy2(input_path, output_path)
        return {
            "fallbackUsed": True,
            "fallbackReason": "svgpathtools is not installed.",
            "diffvgAvailable": diffvg is not None,
            "pathsRemoved": 0,
            "segmentsBefore": 0,
            "segmentsAfter": 0,
        }

    tree = ET.parse(input_path)
    root = tree.getroot()

    removed_paths = 0
    segments_before = 0
    segments_after = 0

    for parent in root.findall(f".//{{{SVG_NS}}}g"):
        children = list(parent)
        for child in children:
            if child.tag != f"{{{SVG_NS}}}path":
                continue

            command = child.get("d")
            if not command:
                parent.remove(child)
                removed_paths += 1
                continue

            path_obj = parse_path(command)
            segments_before += len(path_obj)

            if len(path_obj) == 0 or _estimate_bbox_area(path_obj) < 4.0:
                parent.remove(child)
                removed_paths += 1
                continue

            simplified = _simplify_path(path_obj)
            if len(simplified) == 0:
                parent.remove(child)
                removed_paths += 1
                continue

            segments_after += len(simplified)
            child.set("d", simplified.d())

    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return {
        "fallbackUsed": False,
        "fallbackReason": None,
        "diffvgAvailable": diffvg is not None,
        "pathsRemoved": int(removed_paths),
        "segmentsBefore": int(segments_before),
        "segmentsAfter": int(segments_after),
    }
