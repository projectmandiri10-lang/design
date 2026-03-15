from __future__ import annotations

from datetime import datetime
from pathlib import Path

from export.export_svg import export_svg_file


def export_timestamped_svg(svg_content: str, output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = root / f"vector_{stamp}.svg"
    return export_svg_file(output_path, svg_content)
