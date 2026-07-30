#!/usr/bin/env python3
"""Segment PDFs / images under ~/Internship/graph.

For each PDF ``Foo.pdf`` with N pages creates / overwrites:
  ~/Internship/graph/Foo/
    1/                          — page 1
      region_overlay.png
      info_blocks_00_input.png  — and other debug steps
      ...
    N/
For each image ``Bar.png``:
  ~/Internship/graph/Bar/1/...
Final overlays are also copied to:
  ~/Internship/graph/output/

Usage
-----
python -m edocr2.info_block_segm.run_graph_segm
python -m edocr2.info_block_segm.run_graph_segm knife
python -m edocr2.info_block_segm.run_graph_segm "135-铸造.png"
python ~/Internship/graph/run_graph_segm.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import cv2

_PKG_DIR = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_PKG_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from segm.extract import ExtractConfig, extract_info_blocks
from segm.visualize import save_debug_bundle

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Local project data folder (inputs + batch outputs).
GRAPH_DIR = os.environ.get("IMAGE_SEG_GRAPH", os.path.join(_ROOT, "graph"))
CACHE_DIR = os.path.join(GRAPH_DIR, "_render_cache")
# Flat folder of final overlays only (easy to browse).
OUTPUT_DIR = os.path.join(GRAPH_DIR, "output")

# ---------------------------------------------------------------------------
# Tunable: dilation kernel / 区域边距 (gap)
# ---------------------------------------------------------------------------
# effective_kernel ≈ GAP_THRES × image_width / GAP_REFERENCE_WIDTH
# when SCALE_GAP_BY_RESOLUTION is True.
GAP_THRES = 6
SCALE_GAP_BY_RESOLUTION = True
GAP_REFERENCE_WIDTH = 1200
DILATE_SHAPE = "rect"  # "rect" | "ellipse" | "cross"

# Shared segmentation knobs (applied to every page)
DPI = 200
ALPHA = 0.34
LINE_REMOVAL_MODE = "border_frame"  # "none" | "global" | "border_frame"
BORDER_CLEAR_PX = 80
MIN_AREA = 80
MIN_FILL_RATIO = 0.03
# Keep low-fill outline views if they still have this much ink (px).
MIN_FILL_KEEP_INK = 25000
MAX_AREA_RATIO = 0.35

# Optional OCR / font-height text merge (off by default for batch speed).
TEXT_OCR_REFINE = False
TEXT_GAP_THRES = 10
# ---------------------------------------------------------------------------


def _safe_stem(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")
    return stem or "drawing"


def _find_gs() -> str:
    for candidate in ("gs", "/usr/local/bin/gs", "/opt/homebrew/bin/gs"):
        path = shutil.which(candidate) if os.path.sep not in candidate else candidate
        if path and os.path.isfile(path):
            return path
    raise SystemExit("Ghostscript (gs) not found. Install it or place `gs` on PATH.")


def pdf_page_count(pdf_path: str) -> int:
    """Return number of pages in a PDF."""
    try:
        from PyPDF2 import PdfReader

        n = len(PdfReader(pdf_path).pages)
        if n > 0:
            return int(n)
    except Exception:
        pass

    gs = _find_gs()
    pdf_ps = pdf_path.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    cmd = [
        gs,
        "-q",
        "-dNOSAFER",
        "-dNODISPLAY",
        "-c",
        f"({pdf_ps}) (r) file runpdfbegin pdfpagecount = quit",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    text = (result.stdout or "").strip().splitlines()
    if not text:
        raise SystemExit(f"Could not read page count: {pdf_path}")
    return max(1, int(text[-1].strip()))


def render_pdf_page(
    pdf_path: str,
    png_path: str,
    page: int,
    dpi: int = 200,
) -> str:
    """Render one 1-based PDF page without annotation balloons."""
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    gs = _find_gs()
    page = max(1, int(page))
    cmd = [
        gs,
        "-q",
        "-dNOSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        f"-dFirstPage={page}",
        f"-dLastPage={page}",
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
        raise SystemExit(f"Failed to render PDF page {page}: {pdf_path}")
    return png_path


def collect_final_overlay(stem: str, page: int, overlay_path: Optional[str]) -> Optional[str]:
    """Copy final region_overlay into GRAPH_DIR/output/ for quick review."""
    if not overlay_path or not os.path.isfile(overlay_path):
        return None
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dest = os.path.join(OUTPUT_DIR, f"{stem}_p{page}.png")
    shutil.copy2(overlay_path, dest)
    return dest


def clear_output_dir(out_dir: str) -> None:
    """Remove previous demo outputs so this run fully overwrites them."""
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        if name.startswith("info_blocks_") or name == "region_overlay.png":
            path = os.path.join(out_dir, name)
            if os.path.isfile(path):
                os.remove(path)


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def list_pdfs(graph_dir: str) -> List[str]:
    if not os.path.isdir(graph_dir):
        raise SystemExit(f"Missing graph folder: {graph_dir}")
    return sorted(
        n
        for n in os.listdir(graph_dir)
        if n.lower().endswith(".pdf") and not n.startswith(".")
    )


def list_images(graph_dir: str) -> List[str]:
    if not os.path.isdir(graph_dir):
        raise SystemExit(f"Missing graph folder: {graph_dir}")
    return sorted(
        n
        for n in os.listdir(graph_dir)
        if n.lower().endswith(_IMAGE_EXTS) and not n.startswith(".")
    )


def apply_work_dir(path: str) -> None:
    """Point GRAPH_DIR / CACHE_DIR / OUTPUT_DIR at a folder (e.g. knife/)."""
    global GRAPH_DIR, CACHE_DIR, OUTPUT_DIR
    work = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(work):
        # Allow relative name under current GRAPH_DIR, e.g. "knife"
        alt = os.path.join(GRAPH_DIR, path)
        if os.path.isdir(alt):
            work = os.path.abspath(alt)
        else:
            raise SystemExit(f"Not a folder: {path}")
    GRAPH_DIR = work
    CACHE_DIR = os.path.join(GRAPH_DIR, "_render_cache")
    OUTPUT_DIR = os.path.join(GRAPH_DIR, "output")


def resolve_inputs(argv: Optional[list]) -> Tuple[List[str], List[str], bool]:
    """Return (pdf_names, image_names, clear_output).

    clear_output is True for a full folder batch (no file args / subfolder arg).
    """
    args = list(argv or [])
    clear_output = False

    if len(args) == 1:
        raw = args[0]
        cand = os.path.expanduser(raw)
        if not os.path.isdir(cand):
            cand = os.path.join(GRAPH_DIR, raw)
        if os.path.isdir(cand):
            apply_work_dir(cand)
            args = []
            clear_output = True

    if not args:
        clear_output = True
        pdfs = list_pdfs(GRAPH_DIR)
        images = list_images(GRAPH_DIR)
        if not pdfs and not images:
            raise SystemExit(f"No PDF/image files found in: {GRAPH_DIR}")
        return pdfs, images, clear_output

    pdfs: List[str] = []
    images: List[str] = []
    for raw in args:
        name = os.path.basename(raw)
        path = os.path.join(GRAPH_DIR, name)
        if not os.path.isfile(path):
            raise SystemExit(f"Not found in graph folder: {name}")
        low = name.lower()
        if low.endswith(".pdf"):
            pdfs.append(name)
        elif low.endswith(_IMAGE_EXTS):
            images.append(name)
        else:
            raise SystemExit(f"Unsupported input type: {name}")
    return pdfs, images, False


def make_config() -> ExtractConfig:
    return ExtractConfig(
        gap_thres=GAP_THRES,
        scale_gap_by_resolution=SCALE_GAP_BY_RESOLUTION,
        gap_reference_width=GAP_REFERENCE_WIDTH,
        dilate_shape=DILATE_SHAPE,
        remove_lines=False,
        line_removal_mode=LINE_REMOVAL_MODE,
        border_line_length_ratio=0.65,
        border_band_ratio=0.04,
        border_clear_px=BORDER_CLEAR_PX,
        re_clear_border_after_dilate=True,
        bridge_break_px=5,
        drop_border_only_blocks=True,
        protect_dense_content=True,
        densify_sparse_components=True,
        min_area=MIN_AREA,
        min_fill_ratio=MIN_FILL_RATIO,
        min_fill_keep_ink=MIN_FILL_KEEP_INK,
        max_area_ratio=MAX_AREA_RATIO,
        max_aspect=0.0,
        group_contained=True,
        container_overlap_frac=0.85,
        text_ocr_refine=TEXT_OCR_REFINE,
        text_gap_thres=TEXT_GAP_THRES,
    )


def _cleanup_pdf_root(pdf_root: str) -> None:
    """Remove stale flat outputs from older single-page runs under pdf_root."""
    os.makedirs(pdf_root, exist_ok=True)
    for name in os.listdir(pdf_root):
        path = os.path.join(pdf_root, name)
        if os.path.isfile(path) and (
            name.startswith("info_blocks_") or name == "region_overlay.png"
        ):
            os.remove(path)


def run_page(
    pdf_name: str,
    page: int,
    n_pages: int,
    config: ExtractConfig,
) -> dict:
    t0 = time.perf_counter()
    pdf_path = os.path.join(GRAPH_DIR, pdf_name)
    stem = _safe_stem(pdf_name)
    pdf_root = os.path.join(GRAPH_DIR, stem)
    out_dir = os.path.join(pdf_root, str(page))
    cache_png = os.path.join(CACHE_DIR, f"{stem}_p{page:02d}_no_annots.png")

    print(f"\n=== {pdf_name}  [{page}/{n_pages}] ===")
    print(f"PDF: {pdf_path}")
    print(f"out: {out_dir}")
    print(f"Rendering page {page} @ {DPI} DPI (annotations off)...")
    t_render0 = time.perf_counter()
    render_pdf_page(pdf_path, cache_png, page=page, dpi=DPI)
    t_render = time.perf_counter() - t_render0

    img = cv2.imread(cache_png)
    if img is None:
        raise SystemExit(f"Failed to read rendered image: {cache_png}")

    clear_output_dir(out_dir)
    print(
        f"Segmenting: gap={config.gap_thres}, "
        f"line_removal_mode={config.line_removal_mode}, "
        f"border_clear_px={config.border_clear_px}"
    )
    t_seg0 = time.perf_counter()
    blocks, debug = extract_info_blocks(img, config=config)
    paths = save_debug_bundle(
        out_dir,
        img,
        blocks,
        debug,
        prefix="info_blocks",
        alpha=ALPHA,
        fill_mode="dilated",
        show_kernel_legend=False,
    )
    t_seg = time.perf_counter() - t_seg0
    elapsed = time.perf_counter() - t0

    print(f"blocks: {len(blocks)}")
    print(f"effective_gap: {debug.get('effective_gap_thres')}")
    print(f"overlay: {paths.get('region_overlay')}")
    final_copy = collect_final_overlay(stem, page, paths.get("region_overlay"))
    if final_copy:
        print(f"output copy: {final_copy}")
    print(
        f"time: total={elapsed:.3f}s "
        f"(render={t_render:.3f}s, segment+save={t_seg:.3f}s)"
    )
    return {
        "name": pdf_name,
        "page": page,
        "n_pages": n_pages,
        "stem": stem,
        "blocks": len(blocks),
        "effective_gap": debug.get("effective_gap_thres"),
        "overlay": paths.get("region_overlay"),
        "output_copy": final_copy,
        "out_dir": out_dir,
        "elapsed": elapsed,
        "t_render": t_render,
        "t_seg": t_seg,
    }


def run_pdf(pdf_name: str, config: ExtractConfig) -> List[dict]:
    pdf_path = os.path.join(GRAPH_DIR, pdf_name)
    stem = _safe_stem(pdf_name)
    pdf_root = os.path.join(GRAPH_DIR, stem)
    _cleanup_pdf_root(pdf_root)

    n_pages = pdf_page_count(pdf_path)
    print(f"\n######## {pdf_name} — {n_pages} page(s) ########")
    results = []
    for page in range(1, n_pages + 1):
        results.append(run_page(pdf_name, page, n_pages, config))

    # Drop leftover page folders if PDF shrank since last run
    keep = {str(i) for i in range(1, n_pages + 1)}
    for name in os.listdir(pdf_root):
        path = os.path.join(pdf_root, name)
        if os.path.isdir(path) and name.isdigit() and name not in keep:
            shutil.rmtree(path)
    return results


def run_image(image_name: str, config: ExtractConfig) -> dict:
    """Segment a single raster image already under GRAPH_DIR (no PDF render)."""
    t0 = time.perf_counter()
    image_path = os.path.join(GRAPH_DIR, image_name)
    stem = _safe_stem(image_name)
    img_root = os.path.join(GRAPH_DIR, stem)
    out_dir = os.path.join(img_root, "1")
    _cleanup_pdf_root(img_root)

    print(f"\n######## {image_name} — image ########")
    print(f"IMG: {image_path}")
    print(f"out: {out_dir}")

    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    clear_output_dir(out_dir)
    print(
        f"Segmenting: gap={config.gap_thres}, "
        f"line_removal_mode={config.line_removal_mode}, "
        f"border_clear_px={config.border_clear_px}"
    )
    t_seg0 = time.perf_counter()
    blocks, debug = extract_info_blocks(img, config=config)
    paths = save_debug_bundle(
        out_dir,
        img,
        blocks,
        debug,
        prefix="info_blocks",
        alpha=ALPHA,
        fill_mode="dilated",
        show_kernel_legend=False,
    )
    t_seg = time.perf_counter() - t_seg0
    elapsed = time.perf_counter() - t0

    print(f"blocks: {len(blocks)}")
    print(f"effective_gap: {debug.get('effective_gap_thres')}")
    print(f"overlay: {paths.get('region_overlay')}")
    final_copy = collect_final_overlay(stem, 1, paths.get("region_overlay"))
    if final_copy:
        print(f"output copy: {final_copy}")
    print(f"time: total={elapsed:.3f}s (segment+save={t_seg:.3f}s)")
    return {
        "name": image_name,
        "page": 1,
        "n_pages": 1,
        "stem": stem,
        "blocks": len(blocks),
        "effective_gap": debug.get("effective_gap_thres"),
        "overlay": paths.get("region_overlay"),
        "output_copy": final_copy,
        "out_dir": out_dir,
        "elapsed": elapsed,
        "t_render": 0.0,
        "t_seg": t_seg,
    }


def main(argv: Optional[list] = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    # resolve_inputs may switch GRAPH_DIR when argv is a subfolder (e.g. knife)
    pdfs, images, clear_output = resolve_inputs(argv)

    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if clear_output and os.path.isdir(OUTPUT_DIR):
        for name in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, name)
            if os.path.isfile(path) and name.lower().endswith(".png"):
                os.remove(path)

    config = make_config()
    print(f"GRAPH_DIR: {GRAPH_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(
        f"Inputs: {len(pdfs)} PDF(s), {len(images)} image(s); "
        f"gap={GAP_THRES}, mode={LINE_REMOVAL_MODE}"
    )

    results: List[dict] = []
    for name in pdfs:
        results.extend(run_pdf(name, config))
    for name in images:
        results.append(run_image(name, config))

    print("\n======= DONE =======")
    total_all = 0.0
    for item in results:
        total_all += item["elapsed"]
        print(
            f"- {item['name']} p{item['page']}/{item['n_pages']}: "
            f"{item['blocks']} blocks, kernel≈{item['effective_gap']}px, "
            f"total={item['elapsed']:.3f}s"
        )
        print(f"  → {item['overlay']}")
        if item.get("output_copy"):
            print(f"  → {item['output_copy']}")
    print(f"\nAll pages total wall time: {total_all:.3f}s")
    print(f"Per-page debug: {GRAPH_DIR}/<stem>/<page>/")
    print(f"Final overlays only: {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
