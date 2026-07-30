#!/usr/bin/env python3
"""Segment the Machining drawing and export each part as a masked rectangular PNG.

Outputs (overwritten every run):
  info_block_debug/challenge/machining/region_overlay.png
  info_block_debug/challenge/machining/parts/part_XXX.png

Usage
-----
python -m edocr2.info_block_segm.export_machining_parts
"""

from __future__ import annotations

import os
import sys
import time

import cv2

_PKG_DIR = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_PKG_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segm.extract import ExtractConfig, extract_info_blocks
from segm.export_parts import export_part_crops
from scripts.run_challenge_segm import (
    CACHE_DIR,
    OUT_ROOT,
    SAMPLE_DIR,
    clear_output_dir,
    render_pdf_no_annots,
)
from segm.visualize import save_debug_bundle

PDF_NAME = "A645AB62K1-Machining.pdf"
OUT_DIR = os.path.join(OUT_ROOT, "machining")
PARTS_DIR = os.path.join(OUT_DIR, "parts")

MACHINING_CFG = ExtractConfig(
    gap_thres=4,
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
)


def main() -> int:
    t_all0 = time.perf_counter()
    pdf_path = os.path.join(SAMPLE_DIR, PDF_NAME)
    if not os.path.isfile(pdf_path):
        raise SystemExit(f"Missing PDF: {pdf_path}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_png = os.path.join(CACHE_DIR, "machining_no_annots.png")

    print("=== Machining parts export ===")
    print(f"PDF: {pdf_path}")
    t_render0 = time.perf_counter()
    render_pdf_no_annots(pdf_path, cache_png, dpi=200)
    t_render = time.perf_counter() - t_render0
    print(f"render: {t_render:.3f}s")

    img = cv2.imread(cache_png)
    if img is None:
        raise SystemExit(f"Failed to read: {cache_png}")

    clear_output_dir(OUT_DIR)

    print(f"segmenting (line_removal_mode={MACHINING_CFG.line_removal_mode})...")
    t_seg0 = time.perf_counter()
    blocks, debug = extract_info_blocks(img, config=MACHINING_CFG)
    t_seg = time.perf_counter() - t_seg0

    t_viz0 = time.perf_counter()
    paths = save_debug_bundle(
        OUT_DIR,
        img,
        blocks,
        debug,
        prefix="info_blocks",
        alpha=0.34,
        fill_mode="dilated",
        show_kernel_legend=False,
    )
    t_viz = time.perf_counter() - t_viz0

    effective_gap = int(debug.get("effective_gap_thres", MACHINING_CFG.gap_thres))
    pad = max(2, effective_gap // 2)

    print(f"exporting parts → {PARTS_DIR}")
    t_exp0 = time.perf_counter()
    manifest = export_part_crops(
        img,
        blocks,
        debug["label_map"],
        PARTS_DIR,
        pad=pad,
        bg_color=(255, 255, 255),
    )
    t_exp = time.perf_counter() - t_exp0
    t_all = time.perf_counter() - t_all0

    print(f"blocks (raw CC kept): {len(blocks)}")
    print(f"parts exported: {len(manifest)}")
    print(f"effective_gap / pad: {effective_gap} / {pad}")
    print(f"line_removal_mode: {debug.get('line_removal_mode')}")
    print(f"overlay: {paths.get('region_overlay')}")
    print(f"parts dir: {PARTS_DIR}")
    print(
        f"time: total={t_all:.3f}s "
        f"(render={t_render:.3f}s, "
        f"segment={t_seg:.3f}s, "
        f"overlay_save={t_viz:.3f}s, "
        f"parts_export={t_exp:.3f}s)"
    )
    print(f"★ 分割处理时间 (PNG→blocks): {t_seg:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
