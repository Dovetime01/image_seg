"""Optional post-pass: merge fragmented text notes via OCR / font-height.

Flow
----
1. Take blocks from the normal pixel pipeline.
2. Mark text-like blocks (OCR word count and/or geometry).
3. Estimate per-block font height (OCR word heights, else CC heights).
4. Cluster blocks with similar font height.
5. Re-dilate *only* ink belonging to each same-font cluster with a larger
   gap (``text_gap_thres``), then replace the original fragments with the
   new connected components.

Non-text blocks (views, frames, …) are left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .extract import (
    InfoBlock,
    ExtractConfig,
    merge_by_dilation,
    resolve_gap_thres,
    _bbox_from_component_on_original,
)


@dataclass
class TextBlockInfo:
    block_id: int
    is_text: bool
    font_height: float
    n_chars: int
    source: str  # "ocr" | "geometry" | "none"


def _ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401

        return True
    except Exception:
        return False


def _tesseract_langs() -> str:
    """Prefer eng+chi_sim when both are installed."""
    try:
        import pytesseract

        available = set(pytesseract.get_languages(config=""))
    except Exception:
        return "eng"
    langs = []
    if "eng" in available:
        langs.append("eng")
    if "chi_sim" in available:
        langs.append("chi_sim")
    return "+".join(langs) if langs else "eng"


def ocr_words(bgr_roi: np.ndarray, lang: Optional[str] = None) -> List[dict]:
    """Run tesseract image_to_data on a BGR crop; return non-empty words."""
    import pytesseract

    if bgr_roi is None or bgr_roi.size == 0:
        return []
    if bgr_roi.ndim == 2:
        rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
    # Upscale tiny crops so tesseract sees readable glyphs.
    h, w = rgb.shape[:2]
    scale = 1.0
    if max(h, w) < 400:
        scale = 400.0 / max(h, w)
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lang = lang or _tesseract_langs()
    conf = f"--psm 6 -l {lang}"
    try:
        data = pytesseract.image_to_data(
            rgb, config=conf, output_type=pytesseract.Output.DICT
        )
    except Exception:
        return []
    words = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf_i = float(data["conf"][i])
        except Exception:
            conf_i = -1.0
        if conf_i >= 0 and conf_i < 35:
            continue
        wh = int(data["height"][i])
        ww = int(data["width"][i])
        if wh < 3 or ww < 2:
            continue
        words.append(
            {
                "text": text,
                "left": int(round(int(data["left"][i]) / scale)),
                "top": int(round(int(data["top"][i]) / scale)),
                "width": int(round(ww / scale)),
                "height": int(round(wh / scale)),
                "conf": conf_i,
            }
        )
    return words


def estimate_font_height_geometry(binary_roi: np.ndarray) -> Optional[float]:
    """Median height of character-sized CCs inside a binary crop."""
    if binary_roi is None or binary_roi.size == 0:
        return None
    roi = binary_roi
    if roi.max() == 0:
        return None
    n, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if n <= 1:
        return None
    rh, rw = roi.shape[:2]
    heights = []
    for i in range(1, n):
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if h < 4 or w < 2:
            continue
        # Reject page-sized junk and hairline noise.
        if h > 0.45 * rh or w > 0.8 * rw:
            continue
        if area < 8:
            continue
        aspect = w / float(max(h, 1))
        if aspect > 8.0:  # long horizontal line fragment
            continue
        heights.append(h)
    if len(heights) < 3:
        return None
    return float(np.median(heights))


def estimate_font_height_ocr(words: Sequence[dict]) -> Optional[float]:
    if not words:
        return None
    heights = [int(w["height"]) for w in words if int(w.get("height", 0)) >= 4]
    if len(heights) < 2:
        return float(heights[0]) if heights else None
    return float(np.median(heights))


def _geometry_text_score(binary_roi: np.ndarray, font_h: Optional[float]) -> float:
    """Cheap 0..1 score: multi-line / text-like structure without OCR."""
    if binary_roi is None or binary_roi.size == 0 or binary_roi.max() == 0:
        return 0.0
    h, w = binary_roi.shape[:2]
    fill = float(np.count_nonzero(binary_roi)) / float(max(1, h * w))
    if fill < 0.01 or fill > 0.55:
        return 0.0
    # Horizontal projection peaks ≈ text lines.
    row_ink = (binary_roi > 0).sum(axis=1).astype(np.float32)
    if row_ink.max() <= 0:
        return 0.0
    thr = 0.15 * float(row_ink.max())
    active = row_ink >= thr
    # Count runs of active rows = rough line count.
    runs = 0
    prev = False
    for v in active:
        if v and not prev:
            runs += 1
        prev = bool(v)
    score = 0.0
    if runs >= 2:
        score += 0.45
    elif runs == 1 and h < 4 * (font_h or h):
        score += 0.2
    if font_h and font_h >= 5:
        expected_lines = h / max(font_h * 1.6, 1.0)
        if expected_lines >= 1.5:
            score += 0.25
        if 0.02 <= fill <= 0.35:
            score += 0.15
    if w > 1.8 * h and runs >= 1:
        score += 0.1
    return float(min(1.0, score))


def _line_ink_ratio(binary_roi: np.ndarray) -> float:
    """Fraction of ink belonging to long/thin line-like CCs (drawing strokes)."""
    if binary_roi is None or binary_roi.size == 0 or binary_roi.max() == 0:
        return 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary_roi, connectivity=8)
    if n <= 1:
        return 0.0
    total = 0
    line = 0
    rh, rw = binary_roi.shape[:2]
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 4:
            continue
        total += area
        aspect = max(bw, bh) / float(max(1, min(bw, bh)))
        # Long rulers / hatching strokes / centerlines.
        if aspect >= 6.0 and max(bw, bh) >= 24:
            line += area
        elif min(bw, bh) <= 4 and max(bw, bh) >= 18:
            line += area
        elif area > 0.02 * rh * rw and aspect >= 3.5:
            line += area
    if total <= 0:
        return 0.0
    return float(line) / float(total)


def _hatch_ink_ratio(binary_roi: np.ndarray) -> float:
    """Fraction of ink in many short mid-aspect strokes (section hatching)."""
    if binary_roi is None or binary_roi.size == 0 or binary_roi.max() == 0:
        return 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary_roi, connectivity=8)
    if n <= 8:
        return 0.0
    total = 0
    hatch = 0
    n_hatch = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 6:
            continue
        total += area
        aspect = max(bw, bh) / float(max(1, min(bw, bh)))
        long_side = max(bw, bh)
        if 12 <= long_side <= 120 and 1.8 <= aspect <= 12.0 and area <= 900:
            hatch += area
            n_hatch += 1
    if total <= 0 or n_hatch < 12:
        return 0.0
    return float(hatch) / float(total)


def _dense_island_count(binary_roi: np.ndarray, min_area: int = 800) -> int:
    """Count separated dense ink islands (e.g. two detail views side by side)."""
    if binary_roi is None or binary_roi.size == 0 or binary_roi.max() == 0:
        return 0
    # Light close to form view bodies, then CC.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, k, iterations=2)
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    count = 0
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            count += 1
    return count


def _char_density(n_chars: int, bbox_area: int) -> float:
    return float(n_chars) / float(max(1, bbox_area)) * 10000.0


def _next_label_id(label_map: np.ndarray) -> int:
    """Next free positive label id for writing into ``label_map``."""
    if label_map is None or label_map.size == 0:
        return 1
    return int(label_map.max()) + 1


def analyze_block_text(
    img: np.ndarray,
    binary: np.ndarray,
    block: InfoBlock,
    cfg: ExtractConfig,
) -> TextBlockInfo:
    """Decide whether a block is text and estimate font height."""
    h_img, w_img = binary.shape[:2]
    x0 = max(0, block.x)
    y0 = max(0, block.y)
    x1 = min(w_img, block.x + block.w)
    y1 = min(h_img, block.y + block.h)
    if x1 <= x0 or y1 <= y0:
        return TextBlockInfo(block.id, False, 0.0, 0, "none")

    bbox_area = max(1, (x1 - x0) * (y1 - y0))
    bbox_ratio = bbox_area / float(max(1, h_img * w_img))
    bw, bh = x1 - x0, y1 - y0
    tall_column = bh > 2.2 * max(bw, 1) and bbox_ratio <= float(cfg.text_max_bbox_ratio) * 2.5
    if bbox_ratio > float(cfg.text_max_bbox_ratio) and not tall_column:
        return TextBlockInfo(block.id, False, 0.0, 0, "none")

    bin_roi = binary[y0:y1, x0:x1]
    fill = float(np.count_nonzero(bin_roi)) / float(max(1, bw * bh))
    # Compact note indices ("11.") have high ink fill but tiny absolute area;
    # do not reject them with the paragraph-scale text_min_ink_area gate.
    compact_glyph = (
        12 <= bw <= 120
        and 14 <= bh <= 100
        and block.area >= 60
        and fill >= 0.12
        and bw <= 3.5 * bh
    )
    if block.area < int(cfg.text_min_ink_area) and not compact_glyph:
        return TextBlockInfo(block.id, False, 0.0, 0, "none")

    geo_h = estimate_font_height_geometry(bin_roi)
    geo_score = _geometry_text_score(bin_roi, geo_h)
    line_ratio = _line_ink_ratio(bin_roi)
    hatch_ratio = _hatch_ink_ratio(bin_roi)
    islands = _dense_island_count(bin_roi)
    max_font = float(cfg.text_max_font_height)
    # Only skip OCR on truly large 2-D regions (not wide single-line note strips).
    skip_ocr = min(bw, bh) > 350 and max(bw, bh) > int(cfg.text_max_crop_side)

    # Single-line strips / note indices: prefer bbox height as font estimate.
    single_line = bh <= max(40, int(1.8 * (geo_h or bh)))
    if (single_line or compact_glyph) and 8 <= bh <= max_font:
        if geo_h is None or geo_h < 0.45 * bh:
            geo_h = float(bh) * 0.9

    words: List[dict] = []
    ocr_h: Optional[float] = None
    n_chars = 0
    if cfg.text_ocr_refine and _ocr_available() and not skip_ocr:
        pad = max(2, int(round(0.02 * max(bw, bh))))
        xa0 = max(0, x0 - pad)
        ya0 = max(0, y0 - pad)
        xa1 = min(w_img if img.ndim == 2 else img.shape[1], x1 + pad)
        ya1 = min(h_img if img.ndim == 2 else img.shape[0], y1 + pad)
        crop = img[ya0:ya1, xa0:xa1] if img.ndim == 2 or img.shape[2] >= 3 else bin_roi
        words = ocr_words(crop)
        ocr_h = estimate_font_height_ocr(words)
        n_chars = sum(len(w["text"]) for w in words)

    font_h = float(ocr_h) if ocr_h and ocr_h > 0 else float(geo_h or 0.0)
    if font_h > max_font:
        if geo_h and 4 <= geo_h <= max_font:
            font_h = float(geo_h)
        elif single_line and 8 <= bh <= max_font:
            font_h = float(bh) * 0.9
        else:
            return TextBlockInfo(block.id, False, 0.0, n_chars, "none")

    # Drawing / hatching / multi-view components must not be text paragraphs.
    dens = _char_density(n_chars, bbox_area)
    drawing_like = (
        line_ratio >= float(cfg.text_max_line_ink_ratio)
        or hatch_ratio >= float(cfg.text_max_hatch_ratio)
        or islands >= 2
    )
    # Digit "1" / thin glyphs look line-like; never reject compact note indices
    # via stroke ratios.
    if compact_glyph:
        drawing_like = False

    # Sparse large boxes with few OCR hits are almost always drawings.
    if skip_ocr and fill < 0.08 and dens < float(cfg.text_min_char_density):
        drawing_like = True
    if bw > 1200 and bh > 500 and dens < float(cfg.text_min_char_density) and n_chars < 30:
        drawing_like = True
    # Wide multi-island (or near-island) mechanical packs: never treat as notes.
    if bw > 900 and bh > 400 and (islands >= 2 or line_ratio >= 0.28 or hatch_ratio >= 0.12):
        drawing_like = True
    if bw > 1500 and bh > 700 and fill < 0.10 and dens < 2.5:
        drawing_like = True
    # Detail views: large sparse bbox + mostly stroke ink / weak char density.
    if (
        bw >= 900
        and bh >= 700
        and fill < 0.08
        and dens < 2.0
        and (line_ratio >= 0.20 or hatch_ratio >= 0.08 or islands >= 1)
    ):
        drawing_like = True

    if drawing_like and dens < float(cfg.text_min_char_density):
        return TextBlockInfo(block.id, False, font_h, n_chars, "drawing")
    if skip_ocr and drawing_like:
        return TextBlockInfo(block.id, False, font_h, n_chars, "drawing")
    # Even with some OCR of dimension numerals, refuse huge hatchy packs.
    if drawing_like and (islands >= 2 or hatch_ratio >= 0.18 or line_ratio >= 0.30) and dens < 4.0:
        return TextBlockInfo(block.id, False, font_h, n_chars, "drawing")
    if drawing_like and bw >= 900 and bh >= 700 and dens < 2.0:
        return TextBlockInfo(block.id, False, font_h, n_chars, "drawing")

    is_text = False
    source = "none"
    if skip_ocr:
        if font_h > 0 and geo_score >= float(cfg.text_geometry_score_min) * 0.8:
            if not drawing_like:
                is_text = True
                source = "geometry_anchor"
    elif n_chars >= int(cfg.text_min_ocr_chars) and font_h > 0:
        if (not drawing_like) or dens >= float(cfg.text_min_char_density):
            is_text = True
            source = "ocr"
    elif geo_score >= float(cfg.text_geometry_score_min) and font_h > 0:
        if not drawing_like:
            is_text = True
            source = "geometry"
    elif n_chars >= max(3, int(cfg.text_min_ocr_chars) // 2) and geo_score >= 0.35:
        if not drawing_like:
            is_text = True
            source = "ocr+geometry"
            if font_h <= 0 and geo_h and geo_h <= max_font:
                font_h = float(geo_h)
    elif single_line and font_h > 0 and n_chars >= 3:
        # Thin note lines with weak geometry score still count as text.
        is_text = True
        source = "ocr_line" if n_chars else "geometry_line"
    elif compact_glyph and font_h > 0 and not drawing_like:
        # High-fill mini glyphs (note indices "11.") — treat as text even when
        # OCR is weak / absolute ink area is below paragraph thresholds.
        is_text = True
        source = "glyph_index"

    return TextBlockInfo(block.id, is_text, font_h, n_chars, source)


def cluster_by_font_height(
    infos: Sequence[TextBlockInfo],
    tol_ratio: float,
    abs_tol: float = 2.0,
) -> List[List[int]]:
    """Union-find clusters of text block ids with similar font height."""
    text_infos = [t for t in infos if t.is_text and t.font_height > 0]
    n = len(text_infos)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        hi = text_infos[i].font_height
        for j in range(i + 1, n):
            hj = text_infos[j].font_height
            denom = max(hi, hj, 1.0)
            if abs(hi - hj) <= max(abs_tol, tol_ratio * denom):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i, info in enumerate(text_infos):
        root = find(i)
        groups.setdefault(root, []).append(info.block_id)
    # Only clusters that can actually merge (≥2 fragments).
    return [ids for ids in groups.values() if len(ids) >= 2]


def cluster_text_by_proximity(
    infos: Sequence[TextBlockInfo],
    blocks_by_id: Dict[int, InfoBlock],
    link_gap: float,
    expand_px: float,
    font_gap_scale: float = 2.8,
) -> List[List[int]]:
    """Merge text fragments whose dilated bboxes already meet — even if font
    height estimates differ (OCR/geometry mismatch on the same paragraph).
    """
    text_infos = [t for t in infos if t.is_text and t.block_id in blocks_by_id]
    text_ids = [t.block_id for t in text_infos]
    font_by_id = {t.block_id: float(t.font_height) for t in text_infos}
    n = len(text_ids)
    if n <= 1:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        bi = blocks_by_id[text_ids[i]]
        for j in range(i + 1, n):
            bj = blocks_by_id[text_ids[j]]
            fa = font_by_id.get(text_ids[i], 0.0)
            fb = font_by_id.get(text_ids[j], 0.0)
            # Kernels already intersect ≈ expanded boxes overlap.
            if _bbox_gap(bi, bj) <= max(0.0, expand_px):
                if _should_link_text(
                    bi,
                    bj,
                    link_gap=max(link_gap, expand_px),
                    font_h_a=fa,
                    font_h_b=fb,
                    font_gap_scale=font_gap_scale,
                ):
                    union(i, j)
                    continue
            if _should_link_text(
                bi,
                bj,
                link_gap=link_gap,
                font_h_a=fa,
                font_h_b=fb,
                font_gap_scale=font_gap_scale,
            ):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i, bid in enumerate(text_ids):
        groups.setdefault(find(i), []).append(bid)
    return [g for g in groups.values() if len(g) >= 2]


def _merge_id_lists(groups: Sequence[Sequence[int]]) -> List[List[int]]:
    """Union overlapping id-lists into disjoint clusters."""
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for g in groups:
        ids = list(g)
        if not ids:
            continue
        for x in ids[1:]:
            union(ids[0], x)
    out: Dict[int, List[int]] = {}
    for g in groups:
        for x in g:
            out.setdefault(find(x), []).append(x)
    # unique preserve
    cleaned = []
    for ids in out.values():
        uniq = sorted(set(ids))
        if len(uniq) >= 2:
            cleaned.append(uniq)
    return cleaned


def _bbox_gap(a: InfoBlock, b: InfoBlock) -> float:
    """Min edge-to-edge distance between two axis-aligned boxes (0 if overlap)."""
    ax1, ay1 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x + b.w, b.y + b.h
    dx = 0
    if ax1 < b.x:
        dx = b.x - ax1
    elif bx1 < a.x:
        dx = a.x - bx1
    dy = 0
    if ay1 < b.y:
        dy = b.y - ay1
    elif by1 < a.y:
        dy = a.y - by1
    if dx == 0 and dy == 0:
        return 0.0
    if dx == 0:
        return float(dy)
    if dy == 0:
        return float(dx)
    return float(np.hypot(dx, dy))


def _x_overlap_frac(a: InfoBlock, b: InfoBlock) -> float:
    overlap = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    return float(overlap) / float(max(1, min(a.w, b.w)))


def _y_overlap_frac(a: InfoBlock, b: InfoBlock) -> float:
    overlap = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    return float(overlap) / float(max(1, min(a.h, b.h)))


def _vertical_gap(a: InfoBlock, b: InfoBlock) -> float:
    ay1, by1 = a.y + a.h, b.y + b.h
    if ay1 < b.y:
        return float(b.y - ay1)
    if by1 < a.y:
        return float(a.y - by1)
    return 0.0


def _horizontal_gap(a: InfoBlock, b: InfoBlock) -> float:
    ax1, bx1 = a.x + a.w, b.x + b.w
    if ax1 < b.x:
        return float(b.x - ax1)
    if bx1 < a.x:
        return float(a.x - bx1)
    return 0.0


def _should_link_text(
    a: InfoBlock,
    b: InfoBlock,
    link_gap: float,
    min_x_overlap: float = 0.25,
    min_y_overlap: float = 0.35,
    font_h_a: float = 0.0,
    font_h_b: float = 0.0,
    font_gap_scale: float = 2.8,
) -> bool:
    """Link note fragments in the same column (or wrapped same-line neighbors).

    Prefer vertical stacking with shared x-range / left-edge alignment so
    adjacent EN / 中文 columns do not cross-wire into broken clusters.
    Horizontal linking is restricted to short line fragments so two detail
    views side-by-side are never merged as "wrapped text".

    Vertical threshold uses the larger of ``link_gap`` and
    ``font_height * font_gap_scale`` so uneven paragraph leading still merges
    when glyph size matches.
    """
    fh = max(float(font_h_a or 0.0), float(font_h_b or 0.0))
    v_gap = float(link_gap)
    if fh >= 6.0:
        # Typical note leading is ~1.2–2.0× glyph; allow up to ~font_gap_scale×
        # so CASTING NOTES item1→item2 still joins across a blank row.
        v_gap = max(v_gap, fh * float(font_gap_scale))
        # Cap so distant unrelated columns do not chain through the page.
        v_gap = min(v_gap, max(float(link_gap) * 1.35, fh * 4.0))

    left_align = abs(a.x - b.x) <= max(30.0, 0.12 * float(min(a.w, b.w)))
    same_col = left_align or _x_overlap_frac(a, b) >= min_x_overlap
    if same_col and _vertical_gap(a, b) <= v_gap:
        return True
    # Wrapped continuation only: both pieces look like single text lines.
    # Never horizontally join tall blocks (detail views / multi-line notes).
    short_a = a.h <= min(120, max(40, int(2.2 * fh))) if fh >= 6 else a.h <= 80
    short_b = b.h <= min(120, max(40, int(2.2 * fh))) if fh >= 6 else b.h <= 80
    if (
        short_a
        and short_b
        and _y_overlap_frac(a, b) >= min_y_overlap
        and _horizontal_gap(a, b) <= min(link_gap, v_gap)
    ):
        return True
    return False


def spatial_subclusters(
    cluster_ids: Sequence[int],
    blocks_by_id: Dict[int, InfoBlock],
    link_gap: float,
    font_by_id: Optional[Dict[int, float]] = None,
    font_gap_scale: float = 2.8,
) -> List[List[int]]:
    """Split a same-font cluster into spatially local column groups."""
    ids = [i for i in cluster_ids if i in blocks_by_id]
    n = len(ids)
    if n <= 1:
        return [ids] if ids else []
    font_by_id = font_by_id or {}
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        bi = blocks_by_id[ids[i]]
        for j in range(i + 1, n):
            bj = blocks_by_id[ids[j]]
            if _should_link_text(
                bi,
                bj,
                link_gap,
                font_h_a=float(font_by_id.get(ids[i], 0.0)),
                font_h_b=float(font_by_id.get(ids[j], 0.0)),
                font_gap_scale=font_gap_scale,
            ):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i, bid in enumerate(ids):
        groups.setdefault(find(i), []).append(bid)
    return [g for g in groups.values() if len(g) >= 2]


def peel_text_columns_from_wide_blocks(
    img: np.ndarray,
    blocks: List[InfoBlock],
    binary: np.ndarray,
    label_map: np.ndarray,
    cfg: ExtractConfig,
) -> Tuple[List[InfoBlock], np.ndarray, List[InfoBlock]]:
    """Carve left text columns out of wide sparse mega-blocks.

    CASTING NOTES item 1 often gets glued into a page-wide sparse band that
    also contains dimension callouts. Peeling the left text strip lets the
    font-height merge reconnect it with notes 2+ below.
    """
    h, w = binary.shape[:2]
    new_map = label_map.astype(np.int32, copy=True)
    peeled: List[InfoBlock] = []
    survivors: List[InfoBlock] = []
    next_id = int(new_map.max()) + 1

    for block in blocks:
        bbox_ratio = (block.w * block.h) / float(max(1, h * w))
        fill = block.area / float(max(1, block.w * block.h))
        # Only operate on wide, sparse bands that begin on the far left
        # (CASTING NOTES glued into a page-spanning spider). Mid-page
        # assemblies often also look "wide+sparse" and must not be peeled.
        if block.x > 0.12 * w:
            survivors.append(block)
            continue
        if block.w < 0.35 * w or fill > 0.09 or bbox_ratio < 0.02:
            survivors.append(block)
            continue
        if block.cc_label <= 0:
            survivors.append(block)
            continue

        # Column mass: find left-most dense vertical strip inside the block.
        # Use a *left-local* threshold (not the global max) so a dense dimension
        # stack on the right cannot drown out the notes column.
        x0, y0 = block.x, block.y
        x1, y1 = block.x + block.w, block.y + block.h
        roi = binary[y0:y1, x0:x1]
        mask = (new_map[y0:y1, x0:x1] == int(block.cc_label)) & (roi > 0)
        if not np.any(mask):
            survivors.append(block)
            continue
        col_ink = mask.sum(axis=0).astype(np.float32)
        if col_ink.max() <= 0:
            survivors.append(block)
            continue
        # Smooth projection to tolerate inter-line gaps in the text column.
        k = 31
        ker = np.ones(k, dtype=np.float32) / float(k)
        smooth = np.convolve(col_ink, ker, mode="same")
        # Search only the left portion of the mega-band for a notes column.
        search_n = min(len(smooth), max(200, int(0.28 * block.w), 1400))
        left_peak = float(smooth[:search_n].max()) if search_n else 0.0
        if left_peak < 25.0:
            survivors.append(block)
            continue
        # Low threshold + large allowed holes so indented / wrapped note lines
        # still count as one column (old logic stopped at ~380px too early).
        thr = max(18.0, 0.10 * left_peak)
        active = smooth[:search_n] >= thr
        start = None
        end = None
        hole = 0
        max_hole = 90
        for i, a in enumerate(active):
            if a:
                if start is None:
                    start = i
                end = i
                hole = 0
            elif start is not None:
                hole += 1
                if hole > max_hole and end is not None and (end - start) >= 120:
                    break
        if start is None or end is None:
            survivors.append(block)
            continue
        # Also extend to the 90th-percentile right edge of ink rows so long
        # note lines are fully captured, not just the densest left cluster.
        row_rights: List[int] = []
        sub = mask[:, start : min(search_n, end + 200)]
        for r in range(sub.shape[0]):
            xs = np.flatnonzero(sub[r])
            if xs.size >= 8:
                row_rights.append(int(xs[-1]) + start)
        if row_rights:
            end = max(end, int(np.percentile(np.asarray(row_rights, dtype=np.float32), 90)))
        end = min(end, search_n - 1)
        if (end - start) < 120:
            survivors.append(block)
            continue
        # Don't peel if the "column" is most of the block width (not a side strip).
        if (end - start) > 0.42 * block.w:
            survivors.append(block)
            continue

        strip_x0 = x0 + start
        strip_x1 = x0 + end + 1
        # Slight pad
        strip_x0 = max(x0, strip_x0 - 4)
        strip_x1 = min(x1, strip_x1 + 4)
        strip_mask = np.zeros((h, w), dtype=bool)
        strip_mask[y0:y1, strip_x0:strip_x1] = (
            new_map[y0:y1, strip_x0:strip_x1] == int(block.cc_label)
        ) & (binary[y0:y1, strip_x0:strip_x1] > 0)
        if int(strip_mask.sum()) < 800:
            survivors.append(block)
            continue

        # Validate strip looks like text.
        ys, xs = np.where(strip_mask)
        sx0, sx1 = int(xs.min()), int(xs.max()) + 1
        sy0, sy1 = int(ys.min()), int(ys.max()) + 1
        # Only carve far-left note columns. Mid-page strips inside assemblies
        # are almost always drawing content.
        if sx0 > 0.15 * w:
            survivors.append(block)
            continue
        probe = InfoBlock(
            id=block.id,
            x=sx0,
            y=sy0,
            w=sx1 - sx0,
            h=sy1 - sy0,
            area=int(strip_mask.sum()),
            cc_label=block.cc_label,
            group_label=block.cc_label,
        )
        info = analyze_block_text(img, binary, probe, cfg)
        strip_bin = np.zeros((sy1 - sy0, sx1 - sx0), dtype=np.uint8)
        strip_bin[strip_mask[sy0:sy1, sx0:sx1]] = 255
        line_r = _line_ink_ratio(strip_bin)
        geo_h = estimate_font_height_geometry(strip_bin)
        geo_s = _geometry_text_score(strip_bin, geo_h)
        # Long strokes without text geometry ⇒ drawing remnant, not notes.
        # (Do NOT use hatch_ratio here: glyph CCs look hatch-like.)
        if line_r >= 0.40 and geo_s < 0.45:
            survivors.append(block)
            continue
        if not info.is_text:
            if (geo_s >= 0.30 and geo_h) or (probe.h >= 200 and probe.w >= 180 and geo_h):
                info = TextBlockInfo(block.id, True, float(geo_h or 20.0), 0, "peeled")
            else:
                survivors.append(block)
                continue

        if not info.is_text:
            survivors.append(block)
            continue

        new_label = next_id
        next_id += 1
        old_label = int(block.cc_label)
        # Relabel the whole column rectangle (dilated halo + ink). Peeling only
        # ink leaves mega-band label residue that still paints over notes in
        # fill_mode="dilated" overlays.
        col_zone = new_map[y0:y1, strip_x0:strip_x1] == old_label
        new_map[y0:y1, strip_x0:strip_x1][col_zone] = new_label

        # Mega-band dilated halo often sticks out left of the ink column. Those
        # no-ink pixels stay on the parent id and show up as a phantom vertical
        # color bar beside the peeled notes — clear / absorb them.
        halo_pad = 48
        xa = max(0, min(strip_x0, sx0) - halo_pad)
        xb = min(w, max(strip_x1, sx1) + halo_pad)
        band = new_map[y0:y1, xa:xb]
        bin_band = binary[y0:y1, xa:xb]
        orphan = (band == old_label) & (bin_band == 0)
        # Prefer attach orphan halo to the peeled notes when it touches them.
        peel_dil = cv2.dilate(
            (band == new_label).astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        absorb = orphan & (peel_dil > 0)
        band[absorb] = new_label
        band[orphan & ~absorb] = 0

        peeled.append(
            InfoBlock(
                id=0,
                x=sx0,
                y=sy0,
                w=sx1 - sx0,
                h=sy1 - sy0,
                area=int(strip_mask.sum()),
                cc_label=new_label,
                group_label=new_label,
            )
        )
        # Keep residual of original block if anything remains.
        residual = new_map == old_label
        if np.any(residual):
            # Drop residual dilated-only fringe left of real residual ink.
            ink_res = residual & (binary > 0)
            if np.any(ink_res):
                ink_x0 = int(np.where(ink_res)[1].min())
                # Only scan a band left of residual ink.
                fx0 = max(0, ink_x0 - 80)
                fringe = residual[:, fx0:ink_x0] & (binary[:, fx0:ink_x0] == 0)
                new_map[:, fx0:ink_x0][fringe] = 0
                residual = new_map == old_label
            if np.any(residual):
                rys, rxs = np.where(residual)
                block.x = int(rxs.min())
                block.y = int(rys.min())
                block.w = int(rxs.max() - block.x + 1)
                block.h = int(rys.max() - block.y + 1)
                block.area = int(np.count_nonzero(residual & (binary > 0)))
                survivors.append(block)
        # else original fully converted to peel

    out = survivors + peeled
    out.sort(key=lambda b: (b.y, b.x))
    for i, b in enumerate(out):
        b.id = i
    return out, new_map, peeled


def merge_font_cluster(
    cluster_ids: Sequence[int],
    blocks_by_id: Dict[int, InfoBlock],
    binary: np.ndarray,
    label_map: np.ndarray,
    effective_text_gap: int,
    dilate_shape: str,
    min_area: int,
) -> Tuple[List[InfoBlock], np.ndarray]:
    """Re-dilate ink of one same-font cluster; return new blocks + updated labels."""
    h, w = binary.shape[:2]
    seed = np.zeros((h, w), dtype=np.uint8)
    old_labels = []
    for bid in cluster_ids:
        block = blocks_by_id.get(bid)
        if block is None:
            continue
        old_labels.append(int(block.cc_label))
        x0, y0 = max(0, block.x), max(0, block.y)
        x1, y1 = min(w, block.x + block.w), min(h, block.y + block.h)
        if x1 <= x0 or y1 <= y0:
            continue
        # Prefer original dilated CC pixels when available.
        if block.cc_label > 0:
            cc = label_map == int(block.cc_label)
            seed[cc] = 255
        roi = binary[y0:y1, x0:x1]
        seed[y0:y1, x0:x1][roi > 0] = 255

    if seed.max() == 0:
        return [], label_map

    dilated = merge_by_dilation(seed, effective_text_gap, dilate_shape)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if n <= 1:
        return [], label_map

    # Clear old labels from the global map, then write new ones.
    # Never overwrite labels that belong to blocks outside this cluster
    # (dilation can expand into neighbors and would orphan their text).
    new_map = label_map.copy()
    old_set = {int(x) for x in old_labels if int(x) > 0}
    for lid in old_set:
        new_map[new_map == lid] = 0

    new_blocks: List[InfoBlock] = []
    next_id = max(int(new_map.max()) + 1, _next_label_id(label_map))
    for lid in range(1, n):
        area_dil = int(stats[lid, cv2.CC_STAT_AREA])
        if area_dil < min_area:
            continue
        x = int(stats[lid, cv2.CC_STAT_LEFT])
        y = int(stats[lid, cv2.CC_STAT_TOP])
        bw = int(stats[lid, cv2.CC_STAT_WIDTH])
        bh = int(stats[lid, cv2.CC_STAT_HEIGHT])
        comp = labels == lid
        # Only claim background / our cleared region — keep foreign labels.
        writable = comp & (new_map == 0)
        if not np.any(writable):
            continue
        # Tight bbox from original ink under the writable dilated region.
        ink = np.zeros_like(binary)
        ink[writable & (binary > 0)] = 255
        if ink.max() == 0:
            continue
        ys, xs = np.where(ink > 0)
        tx, ty = int(xs.min()), int(ys.min())
        tw, th = int(xs.max() - tx + 1), int(ys.max() - ty + 1)
        area = int(np.count_nonzero(ink))
        if area < min_area:
            continue
        new_label = next_id
        next_id += 1
        new_map[writable] = new_label
        new_blocks.append(
            InfoBlock(
                id=0,
                x=tx,
                y=ty,
                w=tw,
                h=th,
                area=area,
                cc_label=new_label,
                group_label=new_label,
            )
        )
    return new_blocks, new_map


def refine_text_blocks(
    img: np.ndarray,
    blocks: List[InfoBlock],
    debug: dict,
    config: Optional[ExtractConfig] = None,
) -> Tuple[List[InfoBlock], dict]:
    """Post-process blocks: merge same-font text fragments with a larger gap.

    No-op when ``config.text_ocr_refine`` is False.
    """
    cfg = config or debug.get("config") or ExtractConfig()
    refine_debug = {
        "enabled": bool(cfg.text_ocr_refine),
        "text_infos": [],
        "clusters": [],
        "merged_from": 0,
        "merged_to": 0,
        "effective_text_gap": 0,
    }
    if not cfg.text_ocr_refine:
        return blocks, refine_debug

    binary = debug.get("binary_nolines")
    if binary is None:
        binary = debug.get("binary")
    if binary is None:
        return blocks, refine_debug

    label_map = debug.get("label_map")
    if label_map is None:
        return blocks, refine_debug
    label_map = label_map.astype(np.int32, copy=True)

    # Peel text columns stuck inside wide sparse mega-blocks (note1 in band).
    blocks, label_map, peeled = peel_text_columns_from_wide_blocks(
        img, blocks, binary, label_map, cfg
    )
    refine_debug["peeled_columns"] = len(peeled)
    debug["label_map"] = label_map

    infos = [analyze_block_text(img, binary, b, cfg) for b in blocks]
    refine_debug["text_infos"] = [
        {
            "id": t.block_id,
            "is_text": t.is_text,
            "font_height": round(t.font_height, 2),
            "n_chars": t.n_chars,
            "source": t.source,
        }
        for t in infos
    ]

    clusters = cluster_by_font_height(
        infos,
        tol_ratio=float(cfg.text_font_height_tol),
        abs_tol=float(cfg.text_font_height_abs_tol),
    )
    refine_debug["font_clusters"] = [list(c) for c in clusters]

    image_width = binary.shape[1]
    effective_text_gap = resolve_gap_thres(
        int(cfg.text_gap_thres),
        image_width=image_width,
        scale_by_resolution=cfg.scale_gap_by_resolution,
        reference_width=cfg.gap_reference_width,
    )
    link_gap = float(effective_text_gap) * float(cfg.text_spatial_link_scale)
    # Same-font notes may have larger leading than the base text gap.
    font_link_gap = float(effective_text_gap) * float(
        getattr(cfg, "text_font_link_scale", 2.2)
    )
    expand_px = float(effective_text_gap)
    refine_debug["effective_text_gap"] = effective_text_gap
    refine_debug["link_gap"] = link_gap
    refine_debug["font_link_gap"] = font_link_gap

    blocks_by_id = {b.id: b for b in blocks}
    font_by_id = {
        t.block_id: float(t.font_height)
        for t in infos
        if t.is_text and t.font_height > 0
    }
    font_gap_scale = float(getattr(cfg, "text_font_link_scale", 2.2))
    prox = cluster_text_by_proximity(
        infos,
        blocks_by_id,
        link_gap=link_gap,
        expand_px=expand_px,
        font_gap_scale=font_gap_scale,
    )
    refine_debug["proximity_clusters"] = [list(c) for c in prox]

    spatial_clusters: List[List[int]] = []
    for cluster in clusters:
        # Same-font cluster uses the looser vertical gap (font-height aware).
        spatial_clusters.extend(
            spatial_subclusters(
                cluster,
                blocks_by_id,
                link_gap=font_link_gap,
                font_by_id=font_by_id,
                font_gap_scale=font_gap_scale,
            )
        )
    spatial_clusters = _merge_id_lists(list(spatial_clusters) + list(prox))
    refine_debug["clusters"] = [list(c) for c in spatial_clusters]

    consumed = set()
    new_merged: List[InfoBlock] = []
    merged_from = 0
    singleton_redilated = 0

    for cluster in spatial_clusters:
        fresh, label_map = merge_font_cluster(
            cluster,
            blocks_by_id,
            binary,
            label_map,
            effective_text_gap=effective_text_gap,
            dilate_shape=cfg.dilate_shape,
            min_area=max(20, int(cfg.min_area)),
        )
        if not fresh:
            continue
        for bid in cluster:
            consumed.add(bid)
        merged_from += len(cluster)
        new_merged.extend(fresh)

    # Chinese (and other) text that is already one contiguous block never enters
    # a ≥2 merge cluster, so it would keep the smaller main-pipeline gap and look
    # "unexpanded" in 07_cc_colors. Re-dilate those singletons with text_gap too.
    for t in infos:
        if not t.is_text or t.block_id in consumed:
            continue
        if t.block_id not in blocks_by_id:
            continue
        fresh, label_map = merge_font_cluster(
            [t.block_id],
            blocks_by_id,
            binary,
            label_map,
            effective_text_gap=effective_text_gap,
            dilate_shape=cfg.dilate_shape,
            min_area=max(20, int(cfg.min_area)),
        )
        if not fresh:
            continue
        consumed.add(t.block_id)
        singleton_redilated += 1
        new_merged.extend(fresh)

    kept = [b for b in blocks if b.id not in consumed]
    # Drop orphans whose label pixels were fully overwritten (should be rare now).
    kept_alive = []
    for b in kept:
        if b.cc_label > 0 and not np.any(label_map == int(b.cc_label)):
            continue
        kept_alive.append(b)
    out = kept_alive + new_merged
    out.sort(key=lambda b: (b.y, b.x))
    for i, b in enumerate(out):
        b.id = i

    refine_debug["merged_from"] = merged_from
    refine_debug["merged_to"] = len(new_merged)
    refine_debug["singleton_redilated"] = singleton_redilated
    debug["label_map"] = label_map
    debug["text_refine"] = refine_debug
    debug["effective_text_gap_thres"] = effective_text_gap
    return out, refine_debug
