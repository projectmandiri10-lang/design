from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def detect_inkscape_binary(explicit_path: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)

    env_value = _env_path("INKSCAPE_BIN")
    if env_value is not None:
        candidates.append(env_value)

    which = shutil.which("inkscape")
    if which:
        candidates.append(Path(which))

    common = [
        Path(r"C:\Program Files\Inkscape\bin\inkscape.exe"),
        Path(r"C:\Program Files\Inkscape\inkscape.exe"),
        Path(r"C:\Program Files (x86)\Inkscape\inkscape.exe"),
    ]
    candidates.extend(common)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _env_path(name: str) -> Path | None:
    import os

    value = os.environ.get(name)
    if not value:
        return None
    return Path(value)


def convert_svg_to_eps_pdf(
    svg_path: str | Path,
    eps_path: str | Path | None,
    pdf_path: str | Path | None,
    inkscape_bin: Path | None,
) -> dict[str, Path | None]:
    source = Path(svg_path)
    if inkscape_bin is None or not inkscape_bin.exists():
        raise FileNotFoundError("Inkscape binary is not configured. Install Inkscape or set INKSCAPE_BIN.")

    result: dict[str, Path | None] = {"eps": None, "pdf": None}
    if eps_path is not None:
        eps_output = Path(eps_path)
        eps_output.parent.mkdir(parents=True, exist_ok=True)
        _run_inkscape_export(inkscape_bin, source, eps_output, "eps")
        result["eps"] = eps_output

    if pdf_path is not None:
        pdf_output = Path(pdf_path)
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        _run_inkscape_export(inkscape_bin, source, pdf_output, "pdf")
        result["pdf"] = pdf_output
    return result


def _run_inkscape_export(binary: Path, source: Path, target: Path, output_type: str) -> None:
    command = [
        str(binary),
        str(source),
        f"--export-type={output_type}",
        f"--export-filename={target}",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
