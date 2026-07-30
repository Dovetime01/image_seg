#!/usr/bin/env python3
"""CLI demo for info-block segmentation (dilate + connected components).

Example
-------
python -m edocr2.info_block_segm.demo \\
    --image 无框/2.png \\
    --gap-thres 8 \\
    --alpha 0.45 \\
    --output-dir info_block_debug
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

# Allow running as script: python scripts/demo.py
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segm.extract import ExtractConfig, extract_info_blocks
from segm.visualize import save_debug_bundle


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract info blocks via dilate+CC")
    p.add_argument("--image", required=True, help="Path to drawing image")
    p.add_argument("--pdf-page", type=int, default=1, help="1-based PDF page")
    p.add_argument("--pdf-dpi", type=int, default=200, help="PDF render DPI")
    p.add_argument(
        "--output-dir",
        default="info_block_debug/current",
        help="Fixed debug folder; every run overwrites files in this folder",
    )
    p.add_argument(
        "--gap-thres",
        type=int,
        default=8,
        help="Proximity merge distance / 区域边距 (px)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Region color overlay strength 0~1 (视频里半透明色块)",
    )
    p.add_argument(
        "--show-kernel-legend",
        action="store_true",
        help="Show kernel size text on overlay (off by default)",
    )
    p.add_argument(
        "--no-scale-gap",
        action="store_true",
        help="Use gap directly on source pixels instead of scaling from canvas width",
    )
    p.add_argument(
        "--gap-reference-width",
        type=int,
        default=1200,
        help="Reference UI/canvas width used to scale gap to source resolution",
    )
    p.add_argument(
        "--fill-mode",
        choices=("dilated", "hull", "pixels", "bbox"),
        default="dilated",
        help="dilated=沿膨胀核边界描色(视频效果), hull=凸包, bbox=矩形",
    )
    p.add_argument(
        "--binary-mode",
        choices=("otsu", "fixed", "adaptive"),
        default="otsu",
        help="Binarization mode",
    )
    p.add_argument("--binary-thres", type=int, default=127, help="Fixed threshold if mode=fixed")
    p.add_argument(
        "--remove-lines",
        action="store_true",
        help="Remove long lines (off by default; title-block lines are informative)",
    )
    p.add_argument("--h-line-min-len", type=int, default=40)
    p.add_argument("--v-line-min-len", type=int, default=40)
    p.add_argument(
        "--border-clear-ratio",
        type=float,
        default=0.006,
        help="Clear outer frame band as ratio of shorter image side",
    )
    p.add_argument(
        "--border-clear-px",
        type=int,
        default=0,
        help="Explicit outer-frame clear width; overrides ratio when >0",
    )
    p.add_argument("--min-area", type=int, default=20)
    p.add_argument("--max-area-ratio", type=float, default=0.60)
    p.add_argument(
        "--max-aspect",
        type=float,
        default=0.0,
        help="Reject long components; <=0 keeps leader/title-block lines",
    )
    p.add_argument(
        "--no-group-contained",
        action="store_true",
        help="Do not group text/components contained by a large view or title block",
    )
    p.add_argument("--container-min-bbox-ratio", type=float, default=0.015)
    p.add_argument("--container-margin", type=int, default=8)
    p.add_argument(
        "--dilate-shape",
        choices=("rect", "ellipse", "cross"),
        default="rect",
        help="Dilation kernel shape",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.image.lower().endswith(".pdf"):
        from pdf2image import convert_from_path
        import numpy as np

        pages = convert_from_path(
            args.image,
            dpi=args.pdf_dpi,
            first_page=args.pdf_page,
            last_page=args.pdf_page,
        )
        if not pages:
            raise SystemExit(
                f"Failed to render PDF page {args.pdf_page}: {args.image}"
            )
        img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    cfg = ExtractConfig(
        binary_mode=args.binary_mode,
        binary_thres=args.binary_thres,
        remove_lines=args.remove_lines,
        h_line_min_len=args.h_line_min_len,
        v_line_min_len=args.v_line_min_len,
        border_clear_ratio=args.border_clear_ratio,
        border_clear_px=args.border_clear_px,
        gap_thres=args.gap_thres,
        scale_gap_by_resolution=not args.no_scale_gap,
        gap_reference_width=args.gap_reference_width,
        dilate_shape=args.dilate_shape,
        min_area=args.min_area,
        max_area_ratio=args.max_area_ratio,
        max_aspect=args.max_aspect,
        group_contained=not args.no_group_contained,
        container_min_bbox_ratio=args.container_min_bbox_ratio,
        container_margin=args.container_margin,
    )

    blocks, debug = extract_info_blocks(img, config=cfg)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    # Keep one stable run directory: remove only files generated by this demo.
    for name in os.listdir(out_dir):
        if name.startswith("info_blocks_") or name == "region_overlay.png":
            path = os.path.join(out_dir, name)
            if os.path.isfile(path):
                os.remove(path)
    paths = save_debug_bundle(
        out_dir,
        img,
        blocks,
        debug,
        prefix="info_blocks",
        alpha=args.alpha,
        fill_mode=args.fill_mode,
        show_kernel_legend=args.show_kernel_legend,
    )

    print(f"image: {args.image}")
    print(f"gap_thres (界面区域边距): {cfg.gap_thres}")
    print(f"effective_gap (原图像素): {debug['effective_gap_thres']}")
    print(f"alpha (颜色强度): {args.alpha}")
    print(f"fill_mode: {args.fill_mode}")
    print(f"blocks: {len(blocks)}")
    print(f"debug dir: {out_dir}")
    print(f"★ 视频风格叠色图: {paths.get('region_overlay')}")
    for name, path in paths.items():
        if name != "region_overlay":
            print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
