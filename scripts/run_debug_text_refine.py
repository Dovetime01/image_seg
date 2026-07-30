#!/usr/bin/env python3
"""Debug runner: OCR text-refine on A014V469 page 1 → graph/debug_text.

Usage
-----
python -m edocr2.info_block_segm.run_debug_text_refine
"""

from __future__ import annotations

import os
import shutil
import sys
import time

import cv2

_PKG_DIR = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_PKG_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segm.extract import ExtractConfig, extract_info_blocks
from scripts.run_graph_segm import (
    GRAPH_DIR,
    render_pdf_page,
    _safe_stem,
)
from segm.visualize import save_debug_bundle

# ---------------------------------------------------------------------------
# Target (this experiment only)
# ---------------------------------------------------------------------------
PDF_NAME = "A014V469_PH_20210402.pdf"
PAGE = 1
OUT_DIR = os.path.join(GRAPH_DIR, "debug_text")
DPI = 200
ALPHA = 0.34

# Same base knobs as run_graph_segm, plus text refine ON.
GAP_THRES = 6
TEXT_OCR_REFINE = True  # master switch for this debug run
TEXT_GAP_THRES = 14  # larger same-font re-merge gap (UI-canvas units)
# ---------------------------------------------------------------------------


def make_config(*, text_refine: bool) -> ExtractConfig:
    return ExtractConfig(
        gap_thres=GAP_THRES,
        scale_gap_by_resolution=True,
        gap_reference_width=1200,
        dilate_shape="rect",
        remove_lines=False,
        line_removal_mode="border_frame",
        border_line_length_ratio=0.65,
        border_band_ratio=0.04,
        border_clear_px=80,
        re_clear_border_after_dilate=True,
        bridge_break_px=5,
        drop_border_only_blocks=True,
        protect_dense_content=True,
        densify_sparse_components=True,
        densify_cut_bridges=True,
        min_area=80,
        min_fill_ratio=0.03,
        min_fill_keep_ink=25000,
        max_area_ratio=0.35,
        max_aspect=0.0,
        group_contained=True,
        container_overlap_frac=0.85,
        text_ocr_refine=text_refine,
        text_gap_thres=TEXT_GAP_THRES,
        text_font_height_tol=0.22,
        text_font_height_abs_tol=2.5,
        text_min_ocr_chars=8,
        text_geometry_score_min=0.45,
        text_max_bbox_ratio=0.08,
        text_min_ink_area=400,
        text_max_crop_side=1600,
        text_max_font_height=80.0,
        text_spatial_link_scale=1.5,
        text_font_link_scale=3.0,
        text_max_line_ink_ratio=0.40,
        text_max_hatch_ratio=0.25,
        text_min_char_density=1.2,
        attach_leader_lines=False,
        leader_min_aspect=8.0,
        leader_min_length=180,
        leader_max_thickness=14,
        leader_attach_gap=80,
    )


def _clear_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def main() -> int:
    pdf_path = os.path.join(GRAPH_DIR, PDF_NAME)
    if not os.path.isfile(pdf_path):
        raise SystemExit(f"Missing PDF: {pdf_path}")

    _clear_dir(OUT_DIR)
    cache_dir = os.path.join(GRAPH_DIR, "_render_cache")
    os.makedirs(cache_dir, exist_ok=True)
    stem = _safe_stem(PDF_NAME)
    cache_png = os.path.join(cache_dir, f"{stem}_p{PAGE:02d}_no_annots.png")

    print(f"PDF: {pdf_path}")
    print(f"page: {PAGE}")
    print(f"OUT: {OUT_DIR}")
    print(f"TEXT_OCR_REFINE={TEXT_OCR_REFINE}, gap={GAP_THRES}, text_gap={TEXT_GAP_THRES}")

    t0 = time.perf_counter()
    render_pdf_page(pdf_path, cache_png, page=PAGE, dpi=DPI)
    img = cv2.imread(cache_png)
    if img is None:
        raise SystemExit(f"Failed to read: {cache_png}")

    # Baseline (no text refine) for side-by-side comparison
    base_dir = os.path.join(OUT_DIR, "baseline_no_text_refine")
    os.makedirs(base_dir, exist_ok=True)
    print("\n--- baseline (text_ocr_refine=False) ---")
    t1 = time.perf_counter()
    blocks0, debug0 = extract_info_blocks(img, config=make_config(text_refine=False))
    paths0 = save_debug_bundle(
        base_dir,
        img,
        blocks0,
        debug0,
        prefix="info_blocks",
        alpha=ALPHA,
        fill_mode="dilated",
        show_kernel_legend=False,
    )
    print(
        f"blocks={len(blocks0)}, effective_gap={debug0.get('effective_gap_thres')}, "
        f"time={time.perf_counter() - t1:.2f}s"
    )
    print(f"overlay: {paths0.get('region_overlay')}")

    # Refined
    ref_dir = os.path.join(OUT_DIR, "with_text_refine")
    os.makedirs(ref_dir, exist_ok=True)
    print(f"\n--- refined (text_ocr_refine={TEXT_OCR_REFINE}) ---")
    t2 = time.perf_counter()
    blocks1, debug1 = extract_info_blocks(
        img, config=make_config(text_refine=TEXT_OCR_REFINE)
    )
    paths1 = save_debug_bundle(
        ref_dir,
        img,
        blocks1,
        debug1,
        prefix="info_blocks",
        alpha=ALPHA,
        fill_mode="dilated",
        show_kernel_legend=False,
    )
    meta = debug1.get("text_refine") or {}
    print(
        f"blocks={len(blocks1)}, effective_gap={debug1.get('effective_gap_thres')}, "
        f"effective_text_gap={meta.get('effective_text_gap')}, "
        f"time={time.perf_counter() - t2:.2f}s"
    )
    print(
        f"text clusters={len(meta.get('clusters') or [])}, "
        f"merged {meta.get('merged_from')} → {meta.get('merged_to')}"
    )
    print(f"overlay: {paths1.get('region_overlay')}")

    # Copy final overlays to debug_text root for quick open
    for src, name in (
        (paths0.get("region_overlay"), "baseline_region_overlay.png"),
        (paths1.get("region_overlay"), "refined_region_overlay.png"),
    ):
        if src and os.path.isfile(src):
            shutil.copy2(src, os.path.join(OUT_DIR, name))

    # Write refine summary
    summary = os.path.join(OUT_DIR, "text_refine_summary.txt")
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"pdf={PDF_NAME}\npage={PAGE}\n")
        f.write(f"gap_thres={GAP_THRES}\ntext_gap_thres={TEXT_GAP_THRES}\n")
        f.write(f"text_ocr_refine={TEXT_OCR_REFINE}\n")
        f.write(f"baseline_blocks={len(blocks0)}\nrefined_blocks={len(blocks1)}\n")
        f.write(f"effective_gap={debug1.get('effective_gap_thres')}\n")
        f.write(f"effective_text_gap={meta.get('effective_text_gap')}\n")
        f.write(f"merged_from={meta.get('merged_from')}\nmerged_to={meta.get('merged_to')}\n")
        f.write(f"clusters={meta.get('clusters')}\n")
        f.write("\ntext_infos:\n")
        for info in meta.get("text_infos") or []:
            f.write(f"  {info}\n")
    print(f"\nsummary: {summary}")
    print(f"total wall: {time.perf_counter() - t0:.2f}s")
    print(f"Quick view: {OUT_DIR}/refined_region_overlay.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
