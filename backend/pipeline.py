"""Load images/PDFs and run segmentation / overlay / export."""

from __future__ import annotations

import io
import os
import zipfile
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from segm.extract import ExtractConfig, extract_info_blocks
from segm.export_parts import export_part_crops
from segm.visualize import colorize_label_map, draw_region_overlay_from_debug

from .store import PageState

ProgressCb = Optional[Callable[[str, str], None]]

# Stable labels for the frontend debug menu
DEBUG_STEP_DEFS = [
    ("input", "00 原图"),
    ("binary", "02 二值化"),
    ("border_cleared", "03 清除边框"),
    ("nolines", "04 去图框线"),
    ("dilated", "05 膨胀合并"),
    ("cc_colors", "07 连通域着色"),
    ("overlay", "08 叠色结果"),
]


def _to_preview_bgr(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        raise ValueError("empty debug array")
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


def build_debug_images(
    img_bgr: np.ndarray,
    blocks,
    debug: dict,
    overlay_bgr: np.ndarray,
) -> Dict[str, bytes]:
    """Encode key pipeline stages as JPEG for the debug menu."""
    out: Dict[str, bytes] = {}
    try:
        out["input"] = encode_jpeg(img_bgr, quality=75)
    except Exception:
        pass
    mapping = {
        "binary": "binary",
        "border_cleared": "binary_border_cleared",
        "nolines": "binary_nolines",
        "dilated": "dilated",
    }
    for step, key in mapping.items():
        arr = debug.get(key)
        if arr is None:
            continue
        try:
            out[step] = encode_jpeg(_to_preview_bgr(arr), quality=75)
        except Exception:
            continue
    lm = debug.get("label_map")
    if lm is not None:
        try:
            out["cc_colors"] = encode_jpeg(colorize_label_map(lm), quality=75)
        except Exception:
            pass
    try:
        out["overlay"] = encode_jpeg(overlay_bgr, quality=80)
    except Exception:
        pass
    return out


# Match challenge machining / pre-refactor CLI for A645-style frames
# (border_clear_px=80 leaves zone letters + outer frame on ~6600px renders).
def default_extract_config(
    gap_thres: int = 6,
    text_ocr_refine: bool = False,
    text_gap_thres: int = 10,
) -> ExtractConfig:
    return ExtractConfig(
        gap_thres=int(gap_thres),
        scale_gap_by_resolution=True,
        gap_reference_width=1200,
        dilate_shape="rect",
        remove_lines=False,
        line_removal_mode="border_frame",
        border_line_length_ratio=0.65,
        border_band_ratio=0.04,
        border_clear_px=140,
        re_clear_border_after_dilate=True,
        bridge_break_px=5,
        drop_border_only_blocks=True,
        protect_dense_content=True,
        densify_sparse_components=True,
        min_area=80,
        # Wireframe views (e.g. Tesla top view) often sit at fill≈0.025 with
        # ink area well below 25k; keep them with a slightly looser gate.
        min_fill_ratio=0.02,
        min_fill_keep_ink=12000,
        max_area_ratio=0.35,
        max_aspect=0.0,
        group_contained=False,
        container_overlap_frac=0.85,
        text_ocr_refine=bool(text_ocr_refine),
        text_gap_thres=int(text_gap_thres),
    )


def encode_png(img_bgr: np.ndarray) -> bytes:
    # Compression 1: much faster encode on large pages; file a bit larger.
    ok, buf = cv2.imencode(".png", img_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return buf.tobytes()


def encode_jpeg(img_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return buf.tobytes()


def overlay_media_type(data: bytes) -> str:
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    return "image/png"


def make_thumbnail(img_bgr: np.ndarray, max_side: int = 240) -> bytes:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(
            img_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return encode_jpeg(img_bgr, quality=70)


def load_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return img


def render_pdf_pages(
    data: bytes,
    dpi: int = 150,
) -> List[np.ndarray]:
    """Render PDF bytes to BGR images (one per page).

    Prefer PyMuPDF (no system poppler). Fall back to pdf2image + poppler.
    Annotations / markups are not rendered (drawing content only).
    """
    errors: List[str] = []

    # 1) PyMuPDF — pure pip wheel, works without poppler
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        zoom = float(dpi) / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        out: List[np.ndarray] = []
        for page in doc:
            # annots=False: skip ink/highlight/stamp comments overlaid on the sheet
            pix = page.get_pixmap(matrix=matrix, alpha=False, annots=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            out.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        doc.close()
        if out:
            return out
        errors.append("PyMuPDF returned 0 pages")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"PyMuPDF: {exc}")

    # 2) pdf2image + poppler (pdftoppm does not expose a clean annots flag;
    #    prefer PyMuPDF above for annotation-free renders)
    try:
        from pdf2image import convert_from_bytes

        pages = convert_from_bytes(data, dpi=dpi, fmt="png")
        out = []
        for pil in pages:
            rgb = np.array(pil.convert("RGB"))
            out.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if out:
            return out
        errors.append("pdf2image returned 0 pages")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pdf2image/poppler: {exc}")

    raise RuntimeError(
        "PDF render failed. Install with: pip install pymupdf "
        "(recommended) or brew install poppler. Details: "
        + " | ".join(errors)
    )


def pages_from_upload(
    filename: str,
    data: bytes,
    pdf_dpi: int = 150,
) -> List[Tuple[str, np.ndarray]]:
    """Return list of (page_name, image_bgr) from one uploaded file."""
    lower = filename.lower()
    stem = os.path.splitext(os.path.basename(filename))[0]
    if lower.endswith(".pdf"):
        images = render_pdf_pages(data, dpi=pdf_dpi)
        return [(f"{stem}_p{i+1}", img) for i, img in enumerate(images)]
    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
        img = load_image_bytes(data)
        return [(stem, img)]
    raise ValueError(f"Unsupported file type: {filename}")


def analyze_page(
    page: PageState,
    gap_thres: int = 6,
    text_ocr_refine: bool = False,
    text_gap_thres: int = 10,
    alpha: float = 0.40,
    progress: ProgressCb = None,
    **extra,
) -> dict:
    """Run segmentation and cache overlay on the page."""
    cfg = default_extract_config(
        gap_thres=gap_thres,
        text_ocr_refine=text_ocr_refine,
        text_gap_thres=text_gap_thres,
    )
    for key, value in extra.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    def _p(step: str, message: str) -> None:
        if progress is not None:
            progress(step, message)

    blocks, debug = extract_info_blocks(
        page.image_bgr,
        config=cfg,
        progress=progress,
    )
    _p("overlay", "正在生成叠色预览…")
    overlay = draw_region_overlay_from_debug(
        page.image_bgr,
        blocks,
        debug,
        alpha=float(alpha),
        fill_mode="dilated",
    )
    page.blocks = blocks
    page.label_map = debug.get("label_map")
    page.binary = debug.get("binary_border_cleared")
    if page.binary is None:
        page.binary = debug.get("binary")
    if page.binary is None:
        page.binary = debug.get("binary_nolines")
    page.overlay_png = encode_png(overlay)
    page.debug_images = build_debug_images(page.image_bgr, blocks, debug, overlay)
    page.analyzed = True
    page.dirty = False
    page.config = cfg
    page.alpha = float(alpha)
    page.undo_stack.clear()
    page.last_meta = {
        "blocks": len(blocks),
        "effective_gap": debug.get("effective_gap_thres"),
        "text_refine": debug.get("text_refine"),
        "gap_thres": cfg.gap_thres,
        "text_ocr_refine": cfg.text_ocr_refine,
        "text_gap_thres": cfg.text_gap_thres,
        "alpha": page.alpha,
        "line_removal_mode": debug.get("line_removal_mode"),
        "border_margin": debug.get("border_margin"),
        "debug_steps": [s for s, _ in DEBUG_STEP_DEFS if s in page.debug_images],
    }
    _p("done", f"分析完成：{len(blocks)} 个区域")
    return page.last_meta


def refresh_overlay(page: PageState, alpha: Optional[float] = None) -> bytes:
    """Rebuild overlay from current label_map / blocks (fast edit path)."""
    if page.label_map is None:
        raise RuntimeError("Page has no label_map; analyze first")
    if alpha is not None:
        page.alpha = float(alpha)
    debug = {"label_map": page.label_map}
    # Keep contours after edit so audit UI matches analyze preview; LUT fill still used.
    overlay = draw_region_overlay_from_debug(
        page.image_bgr,
        page.blocks,
        debug,
        alpha=page.alpha,
        fill_mode="dilated",
        draw_contours=True,
    )
    # JPEG is much faster than PNG on large engineering drawings; preview quality is fine.
    page.overlay_png = encode_jpeg(overlay, quality=88)
    return page.overlay_png


def blocks_to_json(page: PageState) -> List[dict]:
    return [
        {
            "id": b.id,
            "x": b.x,
            "y": b.y,
            "w": b.w,
            "h": b.h,
            "area": b.area,
            "cc_label": b.cc_label,
            "group_label": b.group_label,
        }
        for b in page.blocks
    ]


def export_parts_zip(page: PageState, pad: int = 4) -> bytes:
    if page.label_map is None:
        raise RuntimeError("Analyze the page before exporting parts")
    if not page.blocks:
        raise RuntimeError("当前页没有可导出的组件")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        manifest = export_part_crops(
            page.image_bgr,
            page.blocks,
            page.label_map,
            tmp,
            pad=pad,
            min_bbox_area=500,
            min_mask_area=200,
        )
        if not manifest:
            # Relax thresholds once so small demo pages still export something.
            manifest = export_part_crops(
                page.image_bgr,
                page.blocks,
                page.label_map,
                tmp,
                pad=pad,
                min_bbox_area=50,
                min_mask_area=20,
            )
        if not manifest:
            raise RuntimeError("过滤后没有可导出的组件（区域过小）")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in manifest:
                path = item["path"]
                arc = os.path.basename(path)
                zf.write(path, arcname=arc)
            lines = ["part_idx,group_label,x,y,w,h\n"]
            for item in manifest:
                lines.append(
                    f"{item['part_idx']},{item['group_label']},"
                    f"{item['x']},{item['y']},{item['w']},{item['h']}\n"
                )
            zf.writestr("parts_manifest.txt", "".join(lines))
        return buf.getvalue()


def page_summary(page: PageState) -> dict:
    return {
        "index": page.index,
        "name": page.name,
        "source_file": page.source_file,
        "width": int(page.image_bgr.shape[1]),
        "height": int(page.image_bgr.shape[0]),
        "analyzed": page.analyzed,
        "dirty": page.dirty,
        "block_count": len(page.blocks),
        "can_undo": len(page.undo_stack) > 0,
        "debug_steps": [s for s, _ in DEBUG_STEP_DEFS if s in page.debug_images],
        "meta": page.last_meta,
    }
