#!/usr/bin/env python3
"""Generate info-block segmentation overlays for the Aikedi challenge drawings.

Every run overwrites the previous outputs under:
  info_block_debug/challenge/housing/
  info_block_debug/challenge/machining/
  info_block_debug/challenge/d06/

Usage
-----
python -m edocr2.info_block_segm.run_challenge_segm
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

_PKG_DIR = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_PKG_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segm.extract import ExtractConfig, extract_info_blocks
from segm.visualize import save_debug_bundle

SAMPLE_DIR = os.path.join(_ROOT, "爱柯迪ORC图纸识别参考样例材料")
OUT_ROOT = os.path.join(_ROOT, "info_block_debug", "challenge")
CACHE_DIR = os.path.join(OUT_ROOT, "_render_cache")

# ---------------------------------------------------------------------------
# Tunable: dilation kernel / 区域边距 (gap)
# ---------------------------------------------------------------------------
# gap_thres is the UI spacing slider. With SCALE_GAP_BY_RESOLUTION=True:
#   effective_kernel ≈ gap_thres × image_width / GAP_REFERENCE_WIDTH
# Example: gap=4 on ~6622 px Machining → kernel ≈ 22×22.
#
# Per-drawing gap (edit these):
GAP_HOUSING = 5
GAP_MACHINING = 5
GAP_D06 = 5

# Shared dilation options:
SCALE_GAP_BY_RESOLUTION = True  # False → use gap_thres as raw pixel kernel
GAP_REFERENCE_WIDTH = 1200      # UI canvas width used for scaling
DILATE_SHAPE = "rect"           # "rect" | "ellipse" | "cross"
# ---------------------------------------------------------------------------


@dataclass
class ChallengeJob:
    name: str
    pdf_name: str
    out_subdir: str
    config: ExtractConfig
    alpha: float = 0.34
    dpi: int = 200


JOBS = [
    ChallengeJob(
        name="Housing Casting",
        pdf_name="38213778_006S-HOUSING CASTING, ASSIST (RH) 22MR17.pdf",
        out_subdir="housing",
        config=ExtractConfig(
            gap_thres=GAP_HOUSING,
            scale_gap_by_resolution=SCALE_GAP_BY_RESOLUTION,
            gap_reference_width=GAP_REFERENCE_WIDTH,
            dilate_shape=DILATE_SHAPE,
            remove_lines=False,
            line_removal_mode="none",
            border_clear_px=35,
            min_area=20,
            max_area_ratio=0.6,
            max_aspect=0.0,
            group_contained=True,
        ),
    ),
    ChallengeJob(
        name="Machining Drawing",
        pdf_name="A645AB62K1-Machining.pdf",
        out_subdir="machining",
        config=ExtractConfig(
            gap_thres=GAP_MACHINING,
            scale_gap_by_resolution=SCALE_GAP_BY_RESOLUTION,
            gap_reference_width=GAP_REFERENCE_WIDTH,
            dilate_shape=DILATE_SHAPE,
            remove_lines=False,
            line_removal_mode="border_frame",
            border_line_length_ratio=0.65,
            border_band_ratio=0.04,
            border_clear_px=140,
            re_clear_border_after_dilate=True,
            bridge_break_px=5,
            drop_border_only_blocks=True,
            min_area=150,
            min_fill_ratio=0.03,
            max_area_ratio=0.25,
            max_aspect=0.0,
            group_contained=True,
            container_overlap_frac=0.85,
        ),
    ),
    ChallengeJob(
        name="D06 Drawing",
        pdf_name="0442203384_D_06(1)(1).pdf",
        out_subdir="d06",
        config=ExtractConfig(
            gap_thres=GAP_D06,
            scale_gap_by_resolution=SCALE_GAP_BY_RESOLUTION,
            gap_reference_width=GAP_REFERENCE_WIDTH,
            dilate_shape=DILATE_SHAPE,
            remove_lines=False,
            line_removal_mode="none",
            border_clear_px=50,
            min_area=20,
            max_area_ratio=0.6,
            max_aspect=0.0,
            group_contained=True,
        ),
    ),
]


def _find_gs() -> str:
    for candidate in ("gs", "/usr/local/bin/gs", "/opt/homebrew/bin/gs"):
        path = shutil.which(candidate) if os.path.sep not in candidate else candidate
        if path and os.path.isfile(path):
            return path
    raise SystemExit(
        "Ghostscript (gs) not found. Install it or place `gs` on PATH."
    )


def render_pdf_no_annots(pdf_path: str, png_path: str, dpi: int = 200) -> str:
    """Render first PDF page without annotation balloons."""
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    gs = _find_gs()
    cmd = [
        gs,
        "-q",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-dFirstPage=1",
        "-dLastPage=1",
        "-dShowAnnots=false",
        "-sDEVICE=png16m",
        f"-r{dpi}",
        "-dTextAlphaBits=4",
        "-dGraphicsAlphaBits=4",
        f"-sOutputFile={png_path}",
        pdf_path,
    ]
    subprocess.run(cmd, check=True)
    if not os.path.isfile(png_path):
        raise SystemExit(f"Failed to render PDF: {pdf_path}")
    return png_path


def clear_output_dir(out_dir: str) -> None:
    """Remove previous demo outputs so this run fully overwrites them."""
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if name.startswith("info_blocks_") or name == "region_overlay.png":
            path = os.path.join(out_dir, name)
            if os.path.isfile(path):
                os.remove(path)


def run_one(job: ChallengeJob) -> dict:
    t0 = time.perf_counter()
    pdf_path = os.path.join(SAMPLE_DIR, job.pdf_name)
    if not os.path.isfile(pdf_path):
        raise SystemExit(f"Missing PDF: {pdf_path}")

    cache_png = os.path.join(CACHE_DIR, f"{job.out_subdir}_no_annots.png")
    print(f"\n=== {job.name} ===")
    print(f"PDF: {pdf_path}")
    print(f"Rendering @ {job.dpi} DPI (annotations off)...")
    t_render0 = time.perf_counter()
    render_pdf_no_annots(pdf_path, cache_png, dpi=job.dpi)
    t_render = time.perf_counter() - t_render0

    img = cv2.imread(cache_png)
    if img is None:
        raise SystemExit(f"Failed to read rendered image: {cache_png}")

    out_dir = os.path.join(OUT_ROOT, job.out_subdir)
    clear_output_dir(out_dir)

    print(
        f"Segmenting: gap={job.config.gap_thres}, "
        f"line_removal_mode={job.config.line_removal_mode}, "
        f"border_clear_px={job.config.border_clear_px}"
    )
    t_seg0 = time.perf_counter()
    blocks, debug = extract_info_blocks(img, config=job.config)
    paths = save_debug_bundle(
        out_dir,
        img,
        blocks,
        debug,
        prefix="info_blocks",
        alpha=job.alpha,
        fill_mode="dilated",
        show_kernel_legend=False,
    )
    t_seg = time.perf_counter() - t_seg0
    elapsed = time.perf_counter() - t0

    print(f"blocks: {len(blocks)}")
    print(f"effective_gap: {debug.get('effective_gap_thres')}")
    print(f"line_removal_mode: {debug.get('line_removal_mode')}")
    print(f"overlay: {paths.get('region_overlay')}")
    print(
        f"time: total={elapsed:.3f}s "
        f"(render={t_render:.3f}s, segment+save={t_seg:.3f}s)"
    )
    return {
        "name": job.name,
        "blocks": len(blocks),
        "effective_gap": debug.get("effective_gap_thres"),
        "line_removal_mode": debug.get("line_removal_mode"),
        "overlay": paths.get("region_overlay"),
        "out_dir": out_dir,
        "elapsed": elapsed,
        "t_render": t_render,
        "t_seg": t_seg,
    }


def main(argv: Optional[list] = None) -> int:
    del argv  # reserved for future CLI flags
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    results = []
    for job in JOBS:
        results.append(run_one(job))

    print("\n======= DONE =======")
    total_all = 0.0
    for item in results:
        total_all += item["elapsed"]
        print(
            f"- {item['name']}: {item['blocks']} blocks, "
            f"kernel≈{item['effective_gap']}px, "
            f"mode={item['line_removal_mode']}, "
            f"total={item['elapsed']:.3f}s "
            f"(render={item['t_render']:.3f}s, "
            f"segment+save={item['t_seg']:.3f}s)"
        )
        print(f"  → {item['overlay']}")
    print(f"\nAll drawings total wall time: {total_all:.3f}s")
    print(f"Outputs overwritten under: {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
