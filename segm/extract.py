"""Extract informative ink blocks by dilating foreground then re-labeling CCs.

Pipeline
--------
1. Grayscale + binarize (fixed / Otsu / adaptive)
2. Clear the outer scan/frame border without deleting internal drawing lines
3. Optional long-line removal (disabled by default)
4. Dilate foreground with kernel ≈ gap_thres  → nearby glyphs merge
5. connectedComponentsWithStats on the dilated mask
6. Build each block bbox from the *original* (pre-dilate) foreground inside
   that component, so boxes hug real ink rather than the inflated mask
7. Area / aspect filters

Tuning ``gap_thres`` is equivalent to tuning how far apart two strokes may be
and still count as one info block (boss demo behaviour).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


@dataclass
class InfoBlock:
    """One merged information region."""

    id: int
    x: int
    y: int
    w: int
    h: int
    area: int  # foreground pixel count in original binary (inside bbox)
    # Original connected-component id in dilated label_map (for overlay paint)
    cc_label: int = 0
    # Components inside a large view/title-block container share this label/color.
    group_label: int = 0
    # Optional per-block mask in full-image coordinates (0/255), lazy/optional
    mask: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass
class ExtractConfig:
    """Parameters for info-block extraction."""

    # Binarization
    binary_mode: str = "otsu"  # "fixed" | "otsu" | "adaptive"
    binary_thres: int = 127  # used when binary_mode == "fixed"
    invert: bool = True  # dark ink on light paper → foreground=255

    # Keep drawing/title-block lines: they carry information and connect regions.
    # Enable only when extracting text alone.
    remove_lines: bool = False
    # "none" | "global" | "border_frame"
    # - global: morphology open on whole image (may erase center section lines)
    # - border_frame: only long frame lines on the four image borders
    line_removal_mode: str = "none"
    h_line_min_len: int = 40  # horizontal structuring element length (global mode)
    v_line_min_len: int = 40  # vertical structuring element length (global mode)

    # Border-frame line removal (line_removal_mode="border_frame")
    border_line_length_ratio: float = 0.65  # >= 65% of W/H (0.75 missed some frames)
    border_band_ratio: float = 0.03  # band width as fraction of shorter side
    border_band_px: int = 0  # >0 overrides border_band_ratio
    # Extra top/bottom shelf (fraction of height) to catch title-block separators
    # that sit above the outer frame but still span most of the page width.
    border_span_shelf_ratio: float = 0.15
    border_line_grow_iters: int = 4  # attach adjacent non-text line-like strokes
    # Wipe remaining small zone ticks/letters wholly inside the border band.
    border_band_residual_max_area: int = 8000
    # Protect dense title-block / tolerance-table ink from border cleanup.
    protect_dense_content: bool = True
    protect_density_window: int = 41
    protect_density_thres: float = 0.06
    protect_dilate_px: int = 9

    # Clear only the outer image band before connected components. Engineering
    # drawing frames often touch title blocks and otherwise create one component
    # covering the whole page. 0.006 = 0.6% of the shorter image side.
    border_clear_ratio: float = 0.006
    border_clear_px: int = 0  # >0 overrides border_clear_ratio
    # Re-clear the same outer band on the *dilated* mask so proximity dilate
    # cannot re-bridge the page frame across the cleared margin.
    re_clear_border_after_dilate: bool = True
    # Open the dilated mask to snap hairline bridges between major regions.
    # 0 disables. Typical 3~5 px on source resolution after gap scaling.
    bridge_break_px: int = 0

    # Proximity merge — kernel size ≈ gap threshold (pixels)
    gap_thres: int = 8
    # The UI/video expresses gap on an approximately 1200 px-wide canvas.
    # Scale it for full-resolution drawings (e.g. 8 -> ~31 on a 4600 px image).
    scale_gap_by_resolution: bool = True
    gap_reference_width: int = 1200
    # Dilate shape: "rect" | "ellipse" | "cross"
    dilate_shape: str = "rect"

    # Component filters (applied on original-ink bbox after merge)
    min_area: int = 20
    max_area_ratio: float = 0.35  # reject blobs larger than this fraction of image
    min_w: int = 2
    min_h: int = 2
    max_aspect: float = 0.0  # <=0 disables; long leader/table lines are meaningful
    # Reject sparse wireframe CCs (tiny ink / huge bbox) left by partial frame cuts.
    # Exception: keep if absolute ink area is large enough (thin outline drawings
    # like Top/Front views often have fill≈0.02 but tens of thousands of ink px).
    min_fill_ratio: float = 0.02
    min_fill_keep_ink: int = 25000
    # If a dilated CC is still too sparse, rebuild it from ink (densify) and/or
    # morph-open it so dense islands (e.g. BOM table) are recovered.
    densify_sparse_components: bool = True
    # When densifying sparse CCs, cut page-spanning thin H/V hairlines that
    # look like empty frame separators (lonely middle). Dimension / annotation
    # lines with nearby numeral ink or side ticks are kept.
    densify_cut_bridges: bool = True
    split_sparse_components: bool = False
    sparse_split_open_px: int = 11
    sparse_densify_min_ink: int = 8000
    # After CC: split very tall stacked regions (table / view / notes) at deep
    # horizontal ink valleys so vertical proximity dilate cannot fuse them.
    split_tall_stacked: bool = True
    tall_split_min_height_ratio: float = 0.28
    tall_split_min_height_px: int = 900
    tall_split_min_gap_px: int = 18
    tall_split_valley_ratio: float = 0.12
    tall_split_max_cuts: int = 4
    # Drop blocks whose bbox lies entirely in the outer border band (zone marks).
    drop_border_only_blocks: bool = True
    border_only_band_px: int = 0  # >0 overrides; else uses border_clear / band
    # Merge orphaned note indices ("11.") sitting in the border band onto the
    # same-row text line to their right.
    attach_note_indices: bool = True
    note_index_max_w: int = 120
    note_index_max_h: int = 90
    note_index_max_gap: int = 80
    note_index_min_y_overlap: float = 0.35

    # Group small CCs contained by a large structural CC (view/title block).
    # This fixes title-block text appearing as many unrelated colors while
    # preserving every component's exact dilated outline.
    group_contained: bool = True
    container_min_bbox_ratio: float = 0.015
    container_max_bbox_ratio: float = 0.35
    container_min_fill_ratio: float = 0.04
    container_margin: int = 8
    # Require this fraction of the child bbox area to lie inside the parent.
    # Center-point checks wrongly glue nearby views under a wide table header.
    container_overlap_frac: float = 0.85

    # --- Optional OCR / font-height text merge (post-pass) ---
    # Master switch. When True: mark text-like blocks, cluster by font height,
    # re-dilate same-font clusters with ``text_gap_thres`` (usually > gap_thres).
    text_ocr_refine: bool = False
    # UI-canvas gap for same-font text re-merge (scaled like gap_thres).
    text_gap_thres: int = 10
    # Relative / absolute tolerance to treat two font heights as the same.
    text_font_height_tol: float = 0.22
    text_font_height_abs_tol: float = 2.5
    # OCR: minimum recognized characters to trust a block as text.
    text_min_ocr_chars: int = 8
    # Geometry fallback: minimum text-likeness score in [0, 1].
    text_geometry_score_min: float = 0.45
    # Skip huge view-like boxes / tiny noise.
    text_max_bbox_ratio: float = 0.08
    text_min_ink_area: int = 400
    # Skip OCR/merge on already-large crops (px).
    text_max_crop_side: int = 1600
    # Reject glyph-height estimates above this (px at source resolution).
    text_max_font_height: float = 80.0
    # Spatial link distance = effective_text_gap * this scale (proximity).
    text_spatial_link_scale: float = 1.35
    # Same-font vertical link can be looser (different paragraph leading).
    text_font_link_scale: float = 2.2
    # Drawing vs text: reject blocks whose ink is mostly long thin strokes
    # unless OCR char density is high enough.
    text_max_line_ink_ratio: float = 0.45
    text_max_hatch_ratio: float = 0.25
    text_min_char_density: float = 1.2  # chars per 10k bbox px

    # Attach long thin dimension/leader lines to nearest large component.
    attach_leader_lines: bool = True
    leader_min_aspect: float = 8.0
    leader_min_length: int = 200
    leader_max_thickness: int = 12
    leader_attach_gap: int = 40  # px to parent bbox

    # If True, attach a full-size uint8 mask (expensive); usually False
    return_masks: bool = False


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(
    gray: np.ndarray,
    mode: str = "otsu",
    thres: int = 127,
    invert: bool = True,
) -> np.ndarray:
    """Return uint8 mask with foreground=255, background=0."""
    if mode == "fixed":
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, thres, 255, flag)
    elif mode == "otsu":
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    elif mode == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )
    else:
        raise ValueError(f"Unknown binary_mode: {mode}")
    return binary


def remove_long_lines(
    binary: np.ndarray,
    h_min_len: int = 40,
    v_min_len: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract long horizontal/vertical lines from binary foreground.

    Returns
    -------
    cleaned : binary without long lines
    lines : the removed line mask (for debug)
    """
    h_len = max(1, int(h_min_len))
    v_len = max(1, int(v_min_len))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    lines = cv2.bitwise_or(h_lines, v_lines)
    cleaned = cv2.bitwise_and(binary, cv2.bitwise_not(lines))
    return cleaned, lines


def _border_band_mask(
    shape_hw: Tuple[int, int],
    border_band_ratio: float = 0.03,
    border_band_px: int = 0,
) -> Tuple[np.ndarray, int]:
    h, w = shape_hw
    band = int(border_band_px)
    if band <= 0:
        band = int(round(min(h, w) * max(0.0, float(border_band_ratio))))
    band = min(max(1, band), max(1, min(h, w) // 3))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:band, :] = 255
    mask[h - band :, :] = 255
    mask[:, :band] = 255
    mask[:, w - band :] = 255
    return mask, band


def _span_shelf_mask(
    shape_hw: Tuple[int, int],
    shelf_ratio: float = 0.15,
) -> np.ndarray:
    """Top/bottom shelves for page-spanning title-block separators."""
    h, w = shape_hw
    shelf = int(round(h * max(0.0, float(shelf_ratio))))
    shelf = min(max(0, shelf), max(1, h // 2))
    mask = np.zeros((h, w), dtype=np.uint8)
    if shelf > 0:
        mask[:shelf, :] = 255
        mask[h - shelf :, :] = 255
    return mask


def _collect_long_line_seed(
    binary_roi: np.ndarray,
    min_h_len: int,
    min_v_len: int,
    max_thickness: int,
    axes: str = "hv",
) -> np.ndarray:
    """Morph-open long thin H/V strokes (ROI already AND-masked)."""
    h, w = binary_roi.shape[:2]
    seed = np.zeros((h, w), dtype=np.uint8)
    if not np.any(binary_roi):
        return seed

    if "h" in axes:
        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(40, min(min_h_len, max(40, w // 4))), 1)
        )
        h_lines = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, h_kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(h_lines, connectivity=8)
        for label_id in range(1, n):
            _x, _y, bw, bh, _area = stats[label_id]
            if bw >= min_h_len and bh <= max_thickness:
                seed[labels == label_id] = 255

    if "v" in axes:
        v_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(40, min(min_v_len, max(40, h // 4))))
        )
        v_lines = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, v_kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(v_lines, connectivity=8)
        for label_id in range(1, n):
            _x, _y, bw, bh, _area = stats[label_id]
            if bh >= min_v_len and bw <= max_thickness:
                seed[labels == label_id] = 255
    return seed


def _is_text_like_blob(bw: int, bh: int, area: int) -> bool:
    """Geometry heuristic: compact glyph-like blobs should not be erased."""
    if bw <= 0 or bh <= 0 or area <= 0:
        return False
    aspect = max(bw / bh, bh / bw)
    if aspect > 4.0:
        return False
    fill = area / float(bw * bh)
    if area < 18 or area > 12000:
        return False
    if fill < 0.18:
        return False
    return True


def _is_line_like_blob(bw: int, bh: int, area: int, band: int) -> bool:
    """Thin / elongated strokes attached to frame lines."""
    if bw <= 0 or bh <= 0 or area <= 0:
        return False
    aspect = max(bw / max(bh, 1), bh / max(bw, 1))
    thin = min(bw, bh) <= max(6, band // 4)
    return aspect >= 3.5 or thin


def build_content_protect_mask(
    binary: np.ndarray,
    window: int = 41,
    density_thres: float = 0.06,
    dilate_px: int = 9,
) -> np.ndarray:
    """Mask of dense content (title-block / tolerance tables) that must be kept.

    Long-line / border cleanup must not erase ink under this mask — table text
    sits on top of long grid lines and is otherwise easy collateral damage.
    """
    win = max(3, int(window))
    if win % 2 == 0:
        win += 1
    fg = (binary > 0).astype(np.float32)
    density = cv2.boxFilter(fg, ddepth=-1, ksize=(win, win), normalize=True)
    protect = (density >= float(density_thres)).astype(np.uint8) * 255
    dilate_px = max(0, int(dilate_px))
    if dilate_px > 0:
        if dilate_px % 2 == 0:
            dilate_px += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        protect = cv2.dilate(protect, kernel, iterations=1)
    return protect


def remove_border_frame_lines(
    binary: np.ndarray,
    length_ratio: float = 0.75,
    border_band_ratio: float = 0.03,
    border_band_px: int = 0,
    grow_iters: int = 4,
    protect_mask: Optional[np.ndarray] = None,
    span_shelf_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove long frame lines on borders and title-block separators.

    - Four-edge band: long H and V frame strokes.
    - Extra top/bottom shelf: page-spanning H separators above the title block.
    - Dense protect does **not** block these long thin span seeds (otherwise
      machining frames erase nothing in step 04). Grown residuals still respect
      ``protect_mask``.
    """
    h, w = binary.shape[:2]
    band_mask, band = _border_band_mask(
        (h, w), border_band_ratio=border_band_ratio, border_band_px=border_band_px
    )
    min_h_len = max(1, int(round(w * max(0.0, float(length_ratio)))))
    min_v_len = max(1, int(round(h * max(0.0, float(length_ratio)))))
    max_thickness = max(band * 3, 12)

    border_fg = cv2.bitwise_and(binary, band_mask)
    seed = _collect_long_line_seed(
        border_fg, min_h_len, min_v_len, max_thickness, axes="hv"
    )

    shelf_mask = _span_shelf_mask((h, w), shelf_ratio=span_shelf_ratio)
    shelf_fg = cv2.bitwise_and(binary, shelf_mask)
    shelf_h = _collect_long_line_seed(
        shelf_fg, min_h_len, min_v_len, max_thickness, axes="h"
    )
    seed = cv2.bitwise_or(seed, shelf_h)
    core_seed = seed.copy()

    if not np.any(seed):
        return binary.copy(), seed

    search_mask = cv2.bitwise_or(band_mask, shelf_mask)
    search_fg = cv2.bitwise_and(binary, search_mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(search_fg, connectivity=8)
    remove = seed.copy()
    touch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    max_iters = max(0, int(grow_iters))
    for _ in range(max_iters):
        dilated = cv2.dilate(remove, touch_kernel, iterations=1)
        touch = cv2.bitwise_and(search_fg, dilated)
        candidate_ids = set(int(v) for v in np.unique(labels[touch > 0]) if v > 0)
        added = False
        for label_id in candidate_ids:
            x, y, bw, bh, area = stats[label_id]
            x0, y0, x1, y1 = int(x), int(y), int(x + bw), int(y + bh)
            roi_labels = labels[y0:y1, x0:x1]
            roi_remove = remove[y0:y1, x0:x1]
            comp = roi_labels == label_id
            if np.all(roi_remove[comp] > 0):
                continue
            if protect_mask is not None and np.any(protect_mask[y0:y1, x0:x1][comp] > 0):
                continue
            if _is_text_like_blob(int(bw), int(bh), int(area)):
                continue
            if not _is_line_like_blob(int(bw), int(bh), int(area), band):
                continue
            roi_remove[comp] = 255
            added = True
        if not added:
            break

    if protect_mask is not None:
        grown = cv2.bitwise_and(remove, cv2.bitwise_not(core_seed))
        grown = cv2.bitwise_and(grown, cv2.bitwise_not(protect_mask))
        remove = cv2.bitwise_or(core_seed, grown)

    cleaned = cv2.bitwise_and(binary, cv2.bitwise_not(remove))
    return cleaned, remove


def remove_border_band_residuals(
    binary: np.ndarray,
    border_band_ratio: float = 0.03,
    border_band_px: int = 0,
    max_area: int = 8000,
    protect_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Erase small CCs that live entirely inside the outer border band.

    Targets zone letters/numbers and tick marks. Dense title-block / table
    content under ``protect_mask`` is never erased.
    """
    band_mask, _ = _border_band_mask(
        binary.shape[:2],
        border_band_ratio=border_band_ratio,
        border_band_px=border_band_px,
    )
    border_fg = cv2.bitwise_and(binary, band_mask)
    if not np.any(border_fg):
        return binary.copy(), np.zeros_like(binary)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(border_fg, connectivity=8)
    remove = np.zeros_like(binary)
    max_area = max(0, int(max_area))
    for label_id in range(1, n):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= 0 or area > max_area:
            continue
        x, y, bw, bh, _ = stats[label_id]
        # Keep wide / multi-glyph text strips that sit in the border band
        # (bottom NOTES / QUALITY lines). Zone ticks are short and narrow.
        if bw >= 160 or (bh >= 18 and bw >= 80 and area >= 400):
            continue
        comp = labels[y : y + bh, x : x + bw] == label_id
        if protect_mask is not None and np.any(
            protect_mask[y : y + bh, x : x + bw][comp] > 0
        ):
            continue
        remove[y : y + bh, x : x + bw][comp] = 255

    cleaned = cv2.bitwise_and(binary, cv2.bitwise_not(remove))
    return cleaned, remove


def resolve_line_removal_mode(cfg: "ExtractConfig") -> str:
    mode = (cfg.line_removal_mode or "none").lower()
    if mode not in ("none", "global", "border_frame"):
        raise ValueError(f"Unknown line_removal_mode: {cfg.line_removal_mode}")
    if mode == "none" and cfg.remove_lines:
        return "global"
    return mode


def clear_outer_border(
    binary: np.ndarray,
    border_clear_ratio: float = 0.006,
    border_clear_px: int = 0,
    protect_mask: Optional[np.ndarray] = None,
    restore_interior_min_area: int = 200,
) -> Tuple[np.ndarray, int]:
    """Clear an outer image band to disconnect the page frame.

    Dense title-block content is restored via ``protect_mask``. Components that
    also have substantial ink inside the page (edge rows of the title block)
    are restored so DWG.NO / SCALE text is not snipped with zone ticks.
    """
    h, w = binary.shape[:2]
    margin = int(border_clear_px)
    if margin <= 0:
        margin = int(round(min(h, w) * max(0.0, float(border_clear_ratio))))
    margin = min(max(0, margin), max(0, min(h, w) // 4))
    if margin == 0:
        return binary.copy(), 0

    cleared = binary.copy()
    cleared[:margin, :] = 0
    cleared[h - margin :, :] = 0
    cleared[:, :margin] = 0
    cleared[:, w - margin :] = 0

    if protect_mask is not None:
        restore = (protect_mask > 0) & (binary > 0) & (cleared == 0)
        cleared[restore] = binary[restore]

    lost = (binary > 0) & (cleared == 0)
    if np.any(lost):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (binary > 0).astype(np.uint8) * 255, connectivity=8
        )
        min_area = max(0, int(restore_interior_min_area))
        for lid in range(1, n):
            x, y, bw, bh, area = stats[lid]
            if area <= 0:
                continue
            x1, y1 = int(x + bw), int(y + bh)
            if x >= margin and y >= margin and x1 <= w - margin and y1 <= h - margin:
                continue
            roi = labels[y:y1, x:x1] == lid
            roi_lost = lost[y:y1, x:x1] & roi
            if not np.any(roi_lost):
                continue
            interior = np.zeros_like(roi)
            ya0, ya1 = max(0, margin - y), min(bh, h - margin - y)
            xa0, xa1 = max(0, margin - x), min(bw, w - margin - x)
            if ya1 > ya0 and xa1 > xa0:
                interior[ya0:ya1, xa0:xa1] = roi[ya0:ya1, xa0:xa1]
            interior_area = int(interior.sum())
            if interior_area < min_area:
                continue
            fill = float(area) / float(max(1, bw * bh))
            # Sparse wireframe / page-frame remnants — do not restore.
            if fill < 0.025:
                continue
            if _is_line_like_blob(int(bw), int(bh), int(area), margin):
                if interior_area < max(min_area * 5, 1000):
                    continue
            cleared[y:y1, x:x1][roi_lost] = 255

    return cleared, margin


def _dilate_kernel(gap_thres: int, shape: str) -> np.ndarray:
    """Build dilate kernel. Effective merge distance ≈ gap_thres pixels."""
    k = max(1, int(gap_thres))
    # Odd size keeps the kernel centered on each pixel
    if k % 2 == 0:
        k += 1
    shape = shape.lower()
    if shape == "ellipse":
        morph = cv2.MORPH_ELLIPSE
    elif shape == "cross":
        morph = cv2.MORPH_CROSS
    else:
        morph = cv2.MORPH_RECT
    return cv2.getStructuringElement(morph, (k, k))


def merge_by_dilation(binary: np.ndarray, gap_thres: int, shape: str = "rect") -> np.ndarray:
    """Dilate foreground so glyphs closer than ~gap_thres become one blob."""
    if gap_thres <= 0:
        return binary.copy()
    kernel = _dilate_kernel(gap_thres, shape)
    return cv2.dilate(binary, kernel, iterations=1)


def resolve_gap_thres(
    requested_gap: int,
    image_width: int,
    scale_by_resolution: bool = True,
    reference_width: int = 1200,
) -> int:
    """Convert UI/canvas gap pixels to source-image pixels."""
    requested_gap = max(1, int(requested_gap))
    if not scale_by_resolution:
        return requested_gap
    reference_width = max(1, int(reference_width))
    scale = max(1.0, float(image_width) / reference_width)
    return max(1, int(round(requested_gap * scale)))


def _bbox_from_component_on_original(
    labels: np.ndarray,
    label_id: int,
    original_binary: np.ndarray,
    roi_xywh: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int, int]]:
    """Tight bbox + area using original (pre-dilate) ink inside a dilated CC.

    ``roi_xywh`` is the cheap dilated-component bbox from
    ``connectedComponentsWithStats``; we only search inside that window.
    """
    rx, ry, rw, rh = roi_xywh
    if rw <= 0 or rh <= 0:
        return None
    labels_roi = labels[ry : ry + rh, rx : rx + rw]
    binary_roi = original_binary[ry : ry + rh, rx : rx + rw]
    ink = (labels_roi == label_id) & (binary_roi > 0)
    if not np.any(ink):
        return None
    ys, xs = np.where(ink)
    x0 = int(xs.min()) + rx
    x1 = int(xs.max()) + rx
    y0 = int(ys.min()) + ry
    y1 = int(ys.max()) + ry
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    area = int(ink.sum())
    return x0, y0, w, h, area


def _lonely_frame_bridge_mask(
    ink: np.ndarray,
    work_roi: np.ndarray,
    *,
    orientation: str,
    min_length: int,
    open_kernel: np.ndarray,
    max_thickness: int = 14,
    side_band: int = 28,
) -> np.ndarray:
    """Mask of long thin strokes that are empty frame bridges (safe to erase).

    Cuts page-scale hairlines whose *middle* has no side attachments — typical
    drawing frame / zone separators. Protects long dimension/annotation lines:
    their middle corridor usually has tick marks or numeral ink nearby.
    """
    if ink is None or ink.size == 0 or ink.max() == 0:
        return np.zeros_like(ink) if ink is not None else np.zeros((0, 0), np.uint8)

    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, open_kernel)
    if lines.max() == 0:
        return np.zeros_like(ink)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    cut = np.zeros_like(ink)
    h, w = ink.shape[:2]
    branch_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    line_exclude_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        length = bw if orientation == "h" else bh
        thick = bh if orientation == "h" else bw
        if length < int(min_length) or thick > int(max_thickness):
            continue
        if thick > max(3, int(0.03 * length)):
            continue

        comp = lab == i
        if orientation == "h":
            x0 = x + bw // 3
            x1 = x + (2 * bw) // 3
            mid = np.zeros_like(comp)
            mid[:, x0:x1] = comp[:, x0:x1]
        else:
            y0 = y + bh // 3
            y1 = y + (2 * bh) // 3
            mid = np.zeros_like(comp)
            mid[y0:y1, :] = comp[y0:y1, :]
        if not np.any(mid):
            continue

        mid_u8 = mid.astype(np.uint8) * 255
        # Same-CC branches off the middle (structural joins) → keep.
        near = cv2.dilate(mid_u8, branch_k, iterations=1) > 0
        side = near & (ink > 0) & (~comp)
        if int(np.count_nonzero(side)) >= max(40, length // 80):
            continue

        # Nearby annotation ink in original binary along the middle corridor
        # (dimension numbers / ticks) → keep.
        if orientation == "h":
            y0b = max(0, y - side_band)
            y1b = min(h, y + bh + side_band)
            corridor = work_roi[y0b:y1b, x0:x1] > 0
            line_in = comp[y0b:y1b, x0:x1]
        else:
            x0b = max(0, x - side_band)
            x1b = min(w, x + bw + side_band)
            corridor = work_roi[y0:y1, x0b:x1b] > 0
            line_in = comp[y0:y1, x0b:x1b]
        line_dil = (
            cv2.dilate(line_in.astype(np.uint8) * 255, line_exclude_k, iterations=1) > 0
        )
        ann_area = int(np.count_nonzero(corridor & (~line_dil)))
        if ann_area >= max(60, length // 50):
            continue

        cut[comp] = 255

    return cut


def _is_lonely_skinny_frame_component(
    ink: np.ndarray,
    work_roi: np.ndarray,
    bw: int,
    bh: int,
    min_length: int,
) -> bool:
    """True when a CC is essentially one long lonely frame stroke."""
    thick = min(bw, bh)
    length = max(bw, bh)
    if length < int(min_length) or thick > 16:
        return False
    if thick > max(3, int(0.03 * length)):
        return False
    orientation = "h" if bw >= bh else "v"
    if orientation == "h":
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(min_length, 3), 1))
    else:
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(min_length, 3)))
    cut = _lonely_frame_bridge_mask(
        ink,
        work_roi,
        orientation=orientation,
        min_length=min_length,
        open_kernel=open_k,
    )
    if cut.max() == 0:
        return False
    # Most of the component ink lies on the lonely bridge stroke.
    ink_px = int(np.count_nonzero(ink))
    cut_px = int(np.count_nonzero(cut & (ink > 0)))
    return ink_px > 0 and cut_px >= 0.70 * ink_px


def densify_sparse_dilated_labels(
    labels: np.ndarray,
    work: np.ndarray,
    stats: np.ndarray,
    min_fill_ratio: float = 0.02,
    densify_gap: int = 22,
    min_ink_area: int = 8000,
    min_bbox_area: int = 0,
    density_window: int = 41,
    density_thres: float = 0.05,
    min_island_area: int = 400,
    cut_bridges: bool = True,
) -> np.ndarray:
    """Rebuild sparse wireframe CCs into keepable content blocks.

    The bottom title-block / tolerance tables often form one low-fill spider
    (``fill≈0.02``) glued by page-long hairline frame strokes. That spider is
    visible in ``07_cc_colors`` but dropped by ``min_fill_ratio``, so
    ``region_overlay`` never paints it.

    Fix: cut page-spanning thin H/V bridges inside the sparse ink, then dilate
    the remaining ink and re-label — the title corner becomes one denser block.
    """
    if min_fill_ratio <= 0:
        return labels

    labels = labels.astype(np.int32, copy=True)
    max_id = int(labels.max())
    if max_id <= 0:
        return labels

    densify_gap = max(3, int(densify_gap))
    if densify_gap % 2 == 0:
        densify_gap += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (densify_gap, densify_gap))
    min_bbox_area = max(0, int(min_bbox_area))
    min_ink_area = max(0, int(min_ink_area))
    next_id = max_id + 1
    h_img, w_img = labels.shape[:2]
    # Bridges that span a large fraction of the page (not short table rules).
    bridge_h = max(80, int(0.22 * w_img))
    bridge_v = max(80, int(0.22 * h_img))
    h_bridge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_h, 1))
    v_bridge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge_v))

    for lid in range(1, max_id + 1):
        x0 = int(stats[lid, cv2.CC_STAT_LEFT])
        y0 = int(stats[lid, cv2.CC_STAT_TOP])
        bw = int(stats[lid, cv2.CC_STAT_WIDTH])
        bh = int(stats[lid, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0 or bw * bh < min_bbox_area:
            continue
        pad = densify_gap
        xa = max(0, x0 - pad)
        ya = max(0, y0 - pad)
        xb = min(w_img, x0 + bw + pad)
        yb = min(h_img, y0 + bh + pad)
        roi_labels = labels[ya:yb, xa:xb]
        roi_work = work[ya:yb, xa:xb]
        comp = roi_labels == lid
        if not np.any(comp):
            continue
        ink_area = int(np.count_nonzero(roi_work[comp]))
        fill = ink_area / float(max(1, bw * bh))

        ink = np.zeros(comp.shape, dtype=np.uint8)
        ink[comp & (roi_work > 0)] = 255
        foreign = (roi_labels > 0) & (~comp)

        # Standalone page-frame strokes (high fill in a skinny bbox).
        if cut_bridges and _is_lonely_skinny_frame_component(
            ink,
            roi_work,
            bw,
            bh,
            min_length=bridge_v if bh >= bw else bridge_h,
        ):
            roi_labels[comp] = 0
            continue

        if fill >= min_fill_ratio:
            continue
        if ink_area < min_ink_area:
            roi_labels[comp] = 0
            continue

        # Cut only lonely frame bridges; keep dimension lines that have
        # middle-corridor annotation ink or same-CC side branches.
        broken = ink.copy()
        if cut_bridges:
            if bw >= bridge_h:
                h_cut = _lonely_frame_bridge_mask(
                    broken,
                    roi_work,
                    orientation="h",
                    min_length=bridge_h,
                    open_kernel=h_bridge_k,
                )
                broken[h_cut > 0] = 0
            if bh >= bridge_v:
                v_cut = _lonely_frame_bridge_mask(
                    broken,
                    roi_work,
                    orientation="v",
                    min_length=bridge_v,
                    open_kernel=v_bridge_k,
                )
                broken[v_cut > 0] = 0

        tight = cv2.dilate(broken, kernel, iterations=1)
        # Close gaps inside table/title grids so overlay paints a solid block
        # instead of a hollow wireframe (matches how users read cc_colors).
        close_r = max(densify_gap + 8, 31)
        if close_r % 2 == 0:
            close_r += 1
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_r, close_r))
        tight = cv2.morphologyEx(tight, cv2.MORPH_CLOSE, close_k)
        tight[foreign] = 0
        # Stay inside original sparse footprint (plus small pad already in ROI).
        tight[~comp] = 0
        n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(tight, connectivity=8)
        usable = [
            sid
            for sid in range(1, n2)
            if int(stats2[sid, cv2.CC_STAT_AREA]) >= min_island_area
        ]

        roi_labels[comp] = 0
        if len(usable) >= 2:
            for sid in usable:
                roi_labels[lab2 == sid] = next_id
                next_id += 1
        elif len(usable) == 1:
            roi_labels[lab2 == usable[0]] = lid
        else:
            # Last resort: densify original ink (may still be sparse).
            tight0 = cv2.dilate(ink, kernel, iterations=1)
            tight0[foreign] = 0
            roi_labels[tight0 > 0] = lid

    return labels


def split_sparse_dilated_labels(
    labels: np.ndarray,
    work: np.ndarray,
    stats: np.ndarray,
    min_fill_ratio: float = 0.02,
    open_px: int = 11,
    min_bbox_area: int = 0,
) -> np.ndarray:
    """Break sparse dilated CCs into denser islands via morphological open.

    Hairline bridges create one huge low-fill component that would otherwise be
    discarded (losing the BOM table, etc.). Opening removes the bridges and
    reassigns each remaining island a new label.
    """
    if min_fill_ratio <= 0 or open_px <= 0:
        return labels

    labels = labels.astype(np.int32, copy=True)
    max_id = int(labels.max())
    if max_id <= 0:
        return labels

    open_px = int(open_px)
    if open_px % 2 == 0:
        open_px += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    next_id = max_id + 1
    min_bbox_area = max(0, int(min_bbox_area))

    for lid in range(1, max_id + 1):
        x0 = int(stats[lid, cv2.CC_STAT_LEFT])
        y0 = int(stats[lid, cv2.CC_STAT_TOP])
        bw = int(stats[lid, cv2.CC_STAT_WIDTH])
        bh = int(stats[lid, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue
        if bw * bh < min_bbox_area:
            continue
        roi_labels = labels[y0 : y0 + bh, x0 : x0 + bw]
        roi_work = work[y0 : y0 + bh, x0 : x0 + bw]
        comp = roi_labels == lid
        if not np.any(comp):
            continue
        ink_area = int(np.count_nonzero(roi_work[comp]))
        fill = ink_area / float(max(1, bw * bh))
        if fill >= min_fill_ratio:
            continue

        mask = np.zeros((bh, bw), dtype=np.uint8)
        mask[comp] = 255
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        if n2 <= 2:
            ink = np.zeros_like(mask)
            ink[comp & (roi_work > 0)] = 255
            ink = cv2.dilate(
                ink,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
            if n2 <= 2:
                continue

        roi_labels[comp] = 0
        for sid in range(1, n2):
            if int(stats2[sid, cv2.CC_STAT_AREA]) < 20:
                continue
            roi_labels[lab2 == sid] = next_id
            next_id += 1

    return labels


def split_tall_stacked_components(
    labels: np.ndarray,
    work: np.ndarray,
    stats: np.ndarray,
    min_height_ratio: float = 0.28,
    min_height_px: int = 900,
    min_gap_px: int = 18,
    valley_ratio: float = 0.12,
    max_cuts: int = 4,
) -> np.ndarray:
    """Cut tall vertically-stacked CCs (table→view→notes) at ink valleys.

    Proximity dilate often fuses left-column stacks when gaps are smaller than
    the kernel. A deep, sustained horizontal gap in the original ink is a cheap
    signal to split them back apart.
    """
    labels = labels.astype(np.int32, copy=True)
    h_img, w_img = labels.shape[:2]
    max_id = int(labels.max())
    if max_id <= 0:
        return labels

    min_h = max(int(min_height_px), int(min_height_ratio * h_img))
    min_gap = max(8, int(min_gap_px))
    next_id = max_id + 1
    smooth_k = max(21, (min_gap // 2) * 2 + 1)

    for lid in range(1, max_id + 1):
        x0 = int(stats[lid, cv2.CC_STAT_LEFT])
        y0 = int(stats[lid, cv2.CC_STAT_TOP])
        bw = int(stats[lid, cv2.CC_STAT_WIDTH])
        bh = int(stats[lid, cv2.CC_STAT_HEIGHT])
        if bh < min_h or bw < 80:
            continue
        # Full-page spiders are handled elsewhere; focus on column stacks.
        if bw > 0.55 * w_img and bh > 0.7 * h_img:
            continue

        roi_lab = labels[y0 : y0 + bh, x0 : x0 + bw]
        roi_work = work[y0 : y0 + bh, x0 : x0 + bw]
        comp = roi_lab == lid
        if not np.any(comp):
            continue
        ink = comp & (roi_work > 0)
        row = ink.sum(axis=1).astype(np.float32)
        if float(row.max()) < 30:
            continue
        ker = np.ones(smooth_k, dtype=np.float32) / float(smooth_k)
        sm = np.convolve(row, ker, mode="same")
        gmax = float(sm.max())
        if gmax < 30:
            continue

        # Empty / near-empty horizontal bands with content both above and below.
        low_thr = max(8.0, float(valley_ratio) * gmax)
        low = sm <= low_thr
        cuts: List[int] = []
        run_start = None
        for i, is_low in enumerate(low):
            if is_low and run_start is None:
                run_start = i
            elif (not is_low) and run_start is not None:
                run_end = i
                gap_w = run_end - run_start
                if gap_w >= min_gap:
                    cut = (run_start + run_end) // 2
                    above = float(sm[:run_start].max()) if run_start > 0 else 0.0
                    below = float(sm[run_end:].max()) if run_end < bh else 0.0
                    # Both sides must look like real stacks, not tiny noise.
                    if (
                        above >= 0.25 * gmax
                        and below >= 0.25 * gmax
                        and cut >= int(0.06 * bh)
                        and cut <= int(0.94 * bh)
                        and (not cuts or cut - cuts[-1] >= max(80, int(0.05 * bh)))
                    ):
                        cuts.append(cut)
                        if len(cuts) >= int(max_cuts):
                            break
                run_start = None
        if run_start is not None and len(cuts) < int(max_cuts):
            run_end = bh
            gap_w = run_end - run_start
            if gap_w >= min_gap:
                cut = (run_start + run_end) // 2
                above = float(sm[:run_start].max()) if run_start > 0 else 0.0
                below = float(sm[run_end:].max()) if run_end < bh else 0.0
                if (
                    above >= 0.25 * gmax
                    and below >= 0.25 * gmax
                    and cut >= int(0.06 * bh)
                    and cut <= int(0.94 * bh)
                    and (not cuts or cut - cuts[-1] >= max(80, int(0.05 * bh)))
                ):
                    cuts.append(cut)

        if not cuts:
            continue

        # Partition rows into bands separated by cuts; reassign labels.
        bounds = [0] + cuts + [bh]
        # Keep first band as original lid; later bands get fresh ids.
        for bi in range(1, len(bounds) - 1):
            y_a = bounds[bi]
            y_b = bounds[bi + 1]
            band = comp.copy()
            band[:y_a, :] = False
            band[y_b:, :] = False
            if not np.any(band & ink):
                continue
            # Also take dilated halo in this row range so overlay stays coherent.
            halo = comp.copy()
            halo[:y_a, :] = False
            halo[y_b:, :] = False
            roi_lab[halo] = next_id
            next_id += 1
        # Clear original lid from lower bands (already overwritten) — top band
        # keeps lid where halo still equals lid.

    return labels


def attach_note_indices_to_lines(
    blocks: List[InfoBlock],
    label_map: np.ndarray,
    binary: np.ndarray,
    max_w: int = 120,
    max_h: int = 90,
    max_gap: int = 80,
    min_y_overlap: float = 0.35,
) -> Tuple[List[InfoBlock], np.ndarray]:
    """Attach orphaned note indices (e.g. ``11.``) to the text line on their right.

    Indices near the page border are often their own CC and get dropped or left
    as a tiny separate color while the sentence body is a neighboring block.
    """
    if not blocks:
        return blocks, label_map

    new_map = label_map.astype(np.int32, copy=True)
    h_img, w_img = new_map.shape[:2]
    consumed: set = set()
    by_id = {b.id: b for b in blocks}

    def _y_overlap_frac(a: InfoBlock, b: InfoBlock) -> float:
        ov = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        return float(ov) / float(max(1, min(a.h, b.h)))

    indices = []
    for b in blocks:
        if b.w > int(max_w) or b.h > int(max_h):
            continue
        if b.area < 60 or b.area > 2500:
            continue
        # Prefer left-side / narrow glyph packs (note numbers, not wide captions).
        if b.w > 2.5 * max(b.h, 1) and b.w > 60:
            continue
        if b.x > 0.45 * w_img:
            continue
        indices.append(b)

    for idx in indices:
        best = None
        best_gap = None
        for other in blocks:
            if other.id == idx.id or other.id in consumed:
                continue
            if other.w < 150 or other.h > 120:
                continue  # want a text line, not a huge view
            if other.x + 5 < idx.x + idx.w:
                continue  # must be to the right
            gap = other.x - (idx.x + idx.w)
            if gap < -5 or gap > int(max_gap):
                continue
            if _y_overlap_frac(idx, other) < float(min_y_overlap):
                continue
            if best is None or gap < best_gap:
                best = other
                best_gap = gap
        if best is None:
            continue

        # Reassign index ink+halo onto the host line label.
        host_lab = int(best.cc_label)
        child_lab = int(idx.cc_label)
        if host_lab <= 0 or child_lab <= 0 or host_lab == child_lab:
            continue
        new_map[new_map == child_lab] = host_lab
        # Expand host bbox to include the index.
        x0 = min(best.x, idx.x)
        y0 = min(best.y, idx.y)
        x1 = max(best.x + best.w, idx.x + idx.w)
        y1 = max(best.y + best.h, idx.y + idx.h)
        best.x, best.y = int(x0), int(y0)
        best.w, best.h = int(x1 - x0), int(y1 - y0)
        best.area = int(
            np.count_nonzero((new_map == host_lab) & (binary > 0))
        )
        consumed.add(idx.id)

    if not consumed:
        return blocks, label_map

    out = [b for b in blocks if b.id not in consumed]
    out.sort(key=lambda b: (b.y, b.x))
    for i, b in enumerate(out):
        b.id = i
    return out, new_map


def attach_leader_lines_to_components(
    blocks: List[InfoBlock],
    label_map: np.ndarray,
    binary: np.ndarray,
    min_aspect: float = 8.0,
    min_length: int = 200,
    max_thickness: int = 12,
    attach_gap: int = 40,
    max_area_ratio: float = 0.35,
) -> Tuple[List[InfoBlock], np.ndarray]:
    """Re-assign long thin dimension/leader line CCs to a nearby large component.

    Fixes cases where a long horizontal dimension line under a view is its own
    sparse component (or filtered) while the view itself is a colored region.
    """
    if not blocks:
        return blocks, label_map

    h, w = label_map.shape[:2]
    img_area = max(1, h * w)
    new_map = label_map.astype(np.int32, copy=True)

    def _is_leader(b: InfoBlock) -> bool:
        thick = min(b.w, b.h)
        length = max(b.w, b.h)
        if thick > int(max_thickness):
            return False
        if length < int(min_length):
            return False
        aspect = length / float(max(1, thick))
        if aspect < float(min_aspect):
            return False
        fill = b.area / float(max(1, b.w * b.h))
        return fill < 0.25

    def _is_host(b: InfoBlock) -> bool:
        if _is_leader(b):
            return False
        bbox_ratio = (b.w * b.h) / float(img_area)
        if bbox_ratio < 0.01:
            return False
        if bbox_ratio > float(max_area_ratio):
            return False
        # Prefer solid-ish drawing views over sparse text columns.
        fill = b.area / float(max(1, b.w * b.h))
        return fill >= 0.015 and b.area >= 5000

    hosts = [b for b in blocks if _is_host(b)]
    if not hosts:
        return blocks, new_map
    hosts.sort(key=lambda b: b.w * b.h, reverse=True)

    gap = max(0, int(attach_gap))
    consumed: set = set()
    for block in blocks:
        if not _is_leader(block):
            continue
        best = None
        best_dist = 1e18
        for host in hosts:
            # Expand host bbox slightly.
            hx0, hy0 = host.x - gap, host.y - gap
            hx1, hy1 = host.x + host.w + gap, host.y + host.h + gap
            # Distance from leader bbox to expanded host.
            dx = 0
            if block.x + block.w < hx0:
                dx = hx0 - (block.x + block.w)
            elif block.x > hx1:
                dx = block.x - hx1
            dy = 0
            if block.y + block.h < hy0:
                dy = hy0 - (block.y + block.h)
            elif block.y > hy1:
                dy = block.y - hy1
            dist = float(dx + dy) if (dx == 0 or dy == 0) else float(np.hypot(dx, dy))
            # Prefer hosts that the line is under / beside (same horizontal span).
            x_overlap = max(
                0, min(block.x + block.w, host.x + host.w) - max(block.x, host.x)
            )
            if x_overlap < 0.2 * min(block.w, host.w) and block.w > block.h:
                dist += 50.0
            # Horizontal leaders: prefer a solid view *above* the line, not a
            # flat dimension strip that is itself mostly the line band.
            if block.w > block.h:
                host_cy = host.y + 0.5 * host.h
                line_y = block.y + 0.5 * block.h
                if host_cy > line_y:
                    dist += 200.0  # host mostly below the line — wrong
                host_bottom = host.y + host.h
                if host_bottom <= block.y + gap and host.y < block.y:
                    dist *= 0.35
                host_fill = host.area / float(max(1, host.w * host.h))
                if host.h < 220 and host.w > 2000:
                    dist += 200.0
                if host_fill < 0.04:
                    dist += 150.0
                if host_fill >= 0.04 and host_cy < line_y:
                    dist *= 0.5
            if dist < best_dist:
                best_dist = dist
                best = host
        if best is None or best_dist > gap + 5:
            continue
        # Paint leader dilated pixels into host label.
        lid = int(block.cc_label)
        hid = int(best.group_label or best.cc_label)
        if lid <= 0 or hid <= 0:
            continue
        new_map[new_map == lid] = hid
        # Also claim original ink near the line bbox into host for continuity.
        x0 = max(0, block.x - 2)
        y0 = max(0, block.y - 2)
        x1 = min(w, block.x + block.w + 2)
        y1 = min(h, block.y + block.h + 2)
        roi = binary[y0:y1, x0:x1] > 0
        # Only unlabeled or former leader pixels.
        sub = new_map[y0:y1, x0:x1]
        claim = roi & ((sub == 0) | (sub == hid))
        sub[claim] = hid
        new_map[y0:y1, x0:x1] = sub
        consumed.add(id(block))
        # Expand host bbox to include the line.
        nx0 = min(best.x, block.x)
        ny0 = min(best.y, block.y)
        nx1 = max(best.x + best.w, block.x + block.w)
        ny1 = max(best.y + best.h, block.y + block.h)
        best.x, best.y = nx0, ny0
        best.w, best.h = nx1 - nx0, ny1 - ny0
        best.area = int(best.area + block.area)

    # (b) Orphan long lines filtered out of blocks but present in binary.
    def _nearest_host(x: int, y: int, bw: int, bh: int):
        best = None
        best_dist = 1e18
        for host in hosts:
            hx0, hy0 = host.x - gap, host.y - gap
            hx1, hy1 = host.x + host.w + gap, host.y + host.h + gap
            dx = 0
            if x + bw < hx0:
                dx = hx0 - (x + bw)
            elif x > hx1:
                dx = x - hx1
            dy = 0
            if y + bh < hy0:
                dy = hy0 - (y + bh)
            elif y > hy1:
                dy = y - hy1
            dist = float(dx + dy) if (dx == 0 or dy == 0) else float(np.hypot(dx, dy))
            x_overlap = max(0, min(x + bw, host.x + host.w) - max(x, host.x))
            if bw > bh and x_overlap < 0.15 * min(bw, host.w):
                dist += 80.0
            if bw > bh:
                host_cy = host.y + 0.5 * host.h
                line_y = y + 0.5 * bh
                if host_cy > line_y:
                    dist += 200.0
                if host.y + host.h <= y + gap and host.y < y:
                    dist *= 0.35
                if host.h < 220 and host.w > 2000:
                    dist += 200.0
                fill = host.area / float(max(1, host.w * host.h))
                if fill < 0.04:
                    dist += 150.0
                if fill >= 0.04 and host_cy < line_y:
                    dist *= 0.5
            if dist < best_dist:
                best_dist = dist
                best = host
        if best is None or best_dist > gap + 8:
            return None
        return best

    def _is_leader_wh(bw: int, bh: int, area: int) -> bool:
        thick = min(bw, bh)
        length = max(bw, bh)
        if thick > int(max_thickness) or length < int(min_length):
            return False
        if length / float(max(1, thick)) < float(min_aspect):
            return False
        # Morph-open lines are nearly solid in their thin bbox (fill≈1);
        # accept those by thickness alone.
        if thick <= int(max_thickness) and length >= int(min_length):
            return True
        return (area / float(max(1, bw * bh))) < 0.35

    klen = max(int(min_length), 151)
    if klen % 2 == 0:
        klen += 1
    for shape in ((klen, 1), (1, klen)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, shape)
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
        for i in range(1, n):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not _is_leader_wh(bw, bh, area):
                continue
            comp = lab == i
            vals = new_map[comp]
            nonzero = vals[vals > 0]
            if nonzero.size > 0 and int(np.bincount(nonzero).max()) >= 0.5 * nonzero.size:
                continue
            host = _nearest_host(x, y, bw, bh)
            if host is None:
                continue
            hid = int(host.group_label or host.cc_label)
            if hid <= 0:
                continue
            new_map[comp] = hid
            nx0 = min(host.x, x)
            ny0 = min(host.y, y)
            nx1 = max(host.x + host.w, x + bw)
            ny1 = max(host.y + host.h, y + bh)
            host.x, host.y, host.w, host.h = nx0, ny0, nx1 - nx0, ny1 - ny0
            host.area = int(host.area + area)

    # (c) Peel long H-lines off flat sparse bands onto a denser view above.
    # Initial dilate often glues a dimension line into the annotation strip
    # under a casting view; re-home the line to the view.
    klen = max(int(min_length), 151)
    if klen % 2 == 0:
        klen += 1
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(h_lines, connectivity=8)
    host_by_label = {}
    out = [b for b in blocks if id(b) not in consumed]
    for b in out:
        host_by_label[int(b.group_label or b.cc_label)] = b
        host_by_label[int(b.cc_label)] = b

    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not _is_leader_wh(bw, bh, area) or bw <= bh:
            continue
        comp = lab == i
        vals = new_map[comp]
        nonzero = vals[vals > 0]
        if nonzero.size == 0:
            continue
        cur_lid = int(np.bincount(nonzero).argmax())
        cur_host = host_by_label.get(cur_lid)
        # Only peel from flat / sparse current owners.
        if cur_host is not None:
            cur_fill = cur_host.area / float(max(1, cur_host.w * cur_host.h))
            if not (cur_host.h < 220 or cur_fill < 0.05 or cur_host.w > 3 * cur_host.h):
                continue
        above = None
        best_score = 1e18
        line_y = y + 0.5 * bh
        for host in hosts:
            if host.cc_label == cur_lid or (host.group_label or 0) == cur_lid:
                continue
            host_cy = host.y + 0.5 * host.h
            if host_cy > line_y:
                continue
            if host.y + host.h < y - gap:
                continue
            x_overlap = max(0, min(x + bw, host.x + host.w) - max(x, host.x))
            if x_overlap < 0.25 * bw:
                continue
            fill = host.area / float(max(1, host.w * host.h))
            if fill < 0.03:
                continue
            dist = abs((host.y + host.h) - y)
            score = dist - 100.0 * fill - 0.001 * host.area
            if score < best_score:
                best_score = score
                above = host
        if above is None:
            continue
        hid = int(above.group_label or above.cc_label)
        new_map[comp] = hid
        above.area = int(above.area + area)
        nx0 = min(above.x, x)
        ny0 = min(above.y, y)
        nx1 = max(above.x + above.w, x + bw)
        ny1 = max(above.y + above.h, y + bh)
        above.x, above.y, above.w, above.h = nx0, ny0, nx1 - nx0, ny1 - ny0

    return out, new_map


def extract_info_blocks(
    img: np.ndarray,
    config: Optional[ExtractConfig] = None,
    gap_thres: Optional[int] = None,
    progress: Optional[Callable[[str, str], None]] = None,
    **overrides,
) -> Tuple[List[InfoBlock], dict]:
    """Extract info blocks from a BGR/gray drawing image.

    Parameters
    ----------
    img : HxW or HxWxC uint8 image
    config : ExtractConfig; fields can be overridden by kwargs
    gap_thres : convenience override for config.gap_thres (spacing slider)
    progress : optional ``progress(step_id, message)`` callback for UI status

    Returns
    -------
    blocks : list of InfoBlock sorted top-to-bottom, then left-to-right
    debug : dict of intermediate uint8 images
        - gray, binary, binary_nolines, lines, dilated, labels_vis, overlay prep
    """
    cfg = config or ExtractConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"Unknown ExtractConfig field: {key}")
        setattr(cfg, key, value)
    if gap_thres is not None:
        cfg.gap_thres = gap_thres

    def _p(step: str, message: str) -> None:
        if progress is not None:
            progress(step, message)

    _p("gray", "正在灰度化…")
    gray = _to_gray(img)
    _p("binary", "正在二值化…")
    binary = binarize(gray, cfg.binary_mode, cfg.binary_thres, cfg.invert)

    protect = None
    if cfg.protect_dense_content:
        _p("protect", "正在保护密集内容区…")
        protect = build_content_protect_mask(
            binary,
            window=cfg.protect_density_window,
            density_thres=cfg.protect_density_thres,
            dilate_px=cfg.protect_dilate_px,
        )

    _p("border_clear", "正在清除外边框…")
    border_cleared, border_margin = clear_outer_border(
        binary,
        border_clear_ratio=cfg.border_clear_ratio,
        border_clear_px=cfg.border_clear_px,
        protect_mask=None,  # density protect is for line/residual wipe only
        restore_interior_min_area=200,
    )

    lines = np.zeros_like(binary)
    work = border_cleared
    line_mode = resolve_line_removal_mode(cfg)
    if line_mode == "global":
        _p("line_removal", "正在去除长线…")
        work, lines = remove_long_lines(
            border_cleared, cfg.h_line_min_len, cfg.v_line_min_len
        )
        if protect is not None:
            # Global line wipe is especially dangerous near title-block grids.
            restored = (protect > 0) & (binary > 0)
            work = work.copy()
            work[restored] = binary[restored]
            lines = cv2.bitwise_and(lines, cv2.bitwise_not(protect))
    elif line_mode == "border_frame":
        _p("line_removal", "正在去除图框线…")
        work, lines = remove_border_frame_lines(
            border_cleared,
            length_ratio=cfg.border_line_length_ratio,
            border_band_ratio=cfg.border_band_ratio,
            border_band_px=cfg.border_band_px,
            grow_iters=cfg.border_line_grow_iters,
            protect_mask=protect,
            span_shelf_ratio=cfg.border_span_shelf_ratio,
        )
        residual_clean, residual = remove_border_band_residuals(
            work,
            border_band_ratio=cfg.border_band_ratio,
            border_band_px=cfg.border_band_px,
            max_area=cfg.border_band_residual_max_area,
            protect_mask=protect,
        )
        work = residual_clean
        lines = cv2.bitwise_or(lines, residual)

    effective_gap = resolve_gap_thres(
        cfg.gap_thres,
        image_width=gray.shape[1],
        scale_by_resolution=cfg.scale_gap_by_resolution,
        reference_width=cfg.gap_reference_width,
    )
    _p("dilate", f"正在膨胀合并（核≈{effective_gap}px）…")
    dilated = merge_by_dilation(work, effective_gap, cfg.dilate_shape)
    if cfg.re_clear_border_after_dilate:
        # Do not pass density protect here — restoring dense border ink on the
        # dilated mask re-bridges the page frame into one giant component.
        dilated, _ = clear_outer_border(
            dilated,
            border_clear_ratio=cfg.border_clear_ratio,
            border_clear_px=cfg.border_clear_px,
            protect_mask=None,
            restore_interior_min_area=0,
        )
    if cfg.bridge_break_px and int(cfg.bridge_break_px) > 0:
        _p("bridge_break", "正在断开细桥连接…")
        bb = int(cfg.bridge_break_px)
        if bb % 2 == 0:
            bb += 1
        br_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bb, bb))
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, br_kernel)

    _p("cc", "正在连通域分析…")
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    image_area = gray.shape[0] * gray.shape[1]
    if cfg.densify_sparse_components:
        _p("densify", "正在致密稀疏连通域…")
        labels = densify_sparse_dilated_labels(
            labels,
            work,
            stats,
            min_fill_ratio=cfg.min_fill_ratio,
            densify_gap=max(effective_gap, 15),
            min_ink_area=cfg.sparse_densify_min_ink,
            min_bbox_area=max(10000, int(0.01 * image_area)),
            density_window=cfg.protect_density_window,
            density_thres=max(0.04, cfg.protect_density_thres - 0.01),
            min_island_area=400,
            cut_bridges=bool(cfg.densify_cut_bridges),
        )
        split_mask = np.zeros(dilated.shape, dtype=np.uint8)
        split_mask[labels > 0] = 255
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            split_mask, connectivity=8
        )
    if cfg.split_sparse_components:
        _p("split_sparse", "正在拆分稀疏连通域…")
        labels = split_sparse_dilated_labels(
            labels,
            work,
            stats,
            min_fill_ratio=cfg.min_fill_ratio,
            open_px=cfg.sparse_split_open_px,
            min_bbox_area=max(10000, int(0.02 * image_area)),
        )
        split_mask = np.zeros(dilated.shape, dtype=np.uint8)
        split_mask[labels > 0] = 255
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            split_mask, connectivity=8
        )

    if cfg.split_tall_stacked:
        _p("split_tall", "正在按墨水谷拆分叠层区域…")
        labels = split_tall_stacked_components(
            labels,
            work,
            stats,
            min_height_ratio=float(cfg.tall_split_min_height_ratio),
            min_height_px=int(cfg.tall_split_min_height_px),
            min_gap_px=int(cfg.tall_split_min_gap_px),
            valley_ratio=float(cfg.tall_split_valley_ratio),
            max_cuts=int(cfg.tall_split_max_cuts),
        )
        split_mask = np.zeros(dilated.shape, dtype=np.uint8)
        split_mask[labels > 0] = 255
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            split_mask, connectivity=8
        )

    img_area = image_area
    max_area = int(cfg.max_area_ratio * img_area)
    blocks: List[InfoBlock] = []

    _p("build_blocks", "正在构建信息块…")
    for label_id in range(1, n_labels):  # 0 is background
        # Quick reject using dilated stats (cheap)
        x, y, w, h, dil_area = stats[label_id]
        if dil_area < cfg.min_area:
            continue
        if dil_area > max_area:
            continue

        tight = _bbox_from_component_on_original(
            labels, label_id, work, roi_xywh=(int(x), int(y), int(w), int(h))
        )
        if tight is None:
            continue
        x, y, w, h, area = tight
        if area < cfg.min_area or w < cfg.min_w or h < cfg.min_h:
            continue
        if area > max_area:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if cfg.max_aspect > 0 and aspect > cfg.max_aspect:
            continue
        fill_ratio = area / float(max(1, w * h))
        if (
            cfg.min_fill_ratio > 0
            and fill_ratio < cfg.min_fill_ratio
            and area < max(0, int(cfg.min_fill_keep_ink))
        ):
            continue

        mask = None
        if cfg.return_masks:
            mask = np.zeros(gray.shape[:2], dtype=np.uint8)
            mask[y : y + h, x : x + w][
                (labels[y : y + h, x : x + w] == label_id) & (work[y : y + h, x : x + w] > 0)
            ] = 255

        blocks.append(
            InfoBlock(
                id=label_id,
                x=x,
                y=y,
                w=w,
                h=h,
                area=area,
                cc_label=int(label_id),
                group_label=int(label_id),
                mask=mask,
            )
        )

    if cfg.drop_border_only_blocks and blocks:
        _p("drop_border", "正在过滤边框噪声块…")
        image_h, image_w = gray.shape[:2]
        band = int(cfg.border_only_band_px)
        if band <= 0:
            band = int(border_margin)
            if cfg.line_removal_mode == "border_frame" or line_mode == "border_frame":
                _, band2 = _border_band_mask(
                    (image_h, image_w),
                    border_band_ratio=cfg.border_band_ratio,
                    border_band_px=cfg.border_band_px,
                )
                band = max(band, band2)
        band = max(1, band)

        def _in_border_only(block: InfoBlock) -> bool:
            # Zone ticks are small; keep wide / note-like text in the band
            # (e.g. bottom-of-page NOTES / QUALITY lines).
            if block.w >= 160 and block.area >= 350:
                return False
            if block.w >= 400 and block.area >= 800:
                return False
            if block.h >= 28 and block.w >= 120 and block.area >= 600:
                return False
            if block.h >= 40 and block.w >= 200 and block.area >= 1500:
                return False
            # Note indices like "11." sit in the left band but are real text.
            if (
                15 <= block.w <= 120
                and 18 <= block.h <= 100
                and block.area >= 80
                and block.w <= 2.8 * block.h
            ):
                return False
            return (
                block.x >= 0
                and block.y >= 0
                and block.x + block.w <= image_w
                and block.y + block.h <= image_h
                and (
                    block.x + block.w <= band
                    or block.y + block.h <= band
                    or block.x >= image_w - band
                    or block.y >= image_h - band
                )
            )

        blocks = [b for b in blocks if not _in_border_only(b)]

    blocks.sort(key=lambda b: (b.y, b.x))

    if cfg.group_contained and blocks:
        _p("group", "正在归并包含关系…")
        image_h, image_w = gray.shape[:2]
        image_area = image_h * image_w
        containers = []
        for block in blocks:
            bbox_area = block.w * block.h
            bbox_ratio = bbox_area / max(image_area, 1)
            if bbox_ratio < cfg.container_min_bbox_ratio:
                continue
            if bbox_ratio > cfg.container_max_bbox_ratio:
                continue
            fill = block.area / float(max(1, bbox_area))
            if fill < cfg.container_min_fill_ratio:
                continue
            containers.append(block)
        # Smallest valid enclosing container wins, e.g. title block instead of
        # a larger surrounding structure.
        containers.sort(key=lambda block: block.w * block.h)
        margin = max(0, int(cfg.container_margin))
        overlap_frac = float(cfg.container_overlap_frac)
        for block in blocks:
            child_area = max(1, block.w * block.h)
            for parent in containers:
                if parent.cc_label == block.cc_label:
                    continue
                if parent.w * parent.h <= block.w * block.h:
                    continue
                px0 = parent.x - margin
                py0 = parent.y - margin
                px1 = parent.x + parent.w + margin
                py1 = parent.y + parent.h + margin
                ix0 = max(block.x, px0)
                iy0 = max(block.y, py0)
                ix1 = min(block.x + block.w, px1)
                iy1 = min(block.y + block.h, py1)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                if inter / child_area >= overlap_frac:
                    block.group_label = parent.cc_label
                    break

    # Re-number for readability; keep cc_label for painting dilated regions
    for i, block in enumerate(blocks):
        block.id = i

    if cfg.attach_note_indices and blocks:
        _p("note_index", "正在附着注释序号…")
        blocks, labels = attach_note_indices_to_lines(
            blocks,
            labels.astype(np.int32),
            work,
            max_w=int(cfg.note_index_max_w),
            max_h=int(cfg.note_index_max_h),
            max_gap=int(cfg.note_index_max_gap),
            min_y_overlap=float(cfg.note_index_min_y_overlap),
        )

    if cfg.attach_leader_lines and blocks:
        _p("leader", "正在附着引线…")
        blocks, labels = attach_leader_lines_to_components(
            blocks,
            labels.astype(np.int32),
            work,
            min_aspect=cfg.leader_min_aspect,
            min_length=cfg.leader_min_length,
            max_thickness=cfg.leader_max_thickness,
            attach_gap=cfg.leader_attach_gap,
            max_area_ratio=cfg.max_area_ratio,
        )
        for i, block in enumerate(blocks):
            block.id = i

    debug = {
        "gray": gray,
        "binary": binary,
        "binary_border_cleared": border_cleared,
        "border_margin": border_margin,
        "binary_nolines": work,
        "lines": lines,
        "line_removal_mode": line_mode,
        "protect_mask": protect,
        "dilated": dilated,
        "effective_gap_thres": effective_gap,
        "label_map": labels.astype(np.int32),
        "config": cfg,
    }

    if cfg.text_ocr_refine:
        _p("text_refine", "正在文字二次膨胀…")
        from .text_refine import refine_text_blocks

        blocks, refine_meta = refine_text_blocks(img, blocks, debug, config=cfg)
        debug["text_refine"] = refine_meta

    _p("done_extract", f"分割完成，共 {len(blocks)} 个区域")
    return blocks, debug


def crop_block(img: np.ndarray, block: InfoBlock, pad: int = 2) -> np.ndarray:
    """Axis-aligned crop of a block with optional padding."""
    h, w = img.shape[:2]
    x0 = max(0, block.x - pad)
    y0 = max(0, block.y - pad)
    x1 = min(w, block.x + block.w + pad)
    y1 = min(h, block.y + block.h + pad)
    return img[y0:y1, x0:x1].copy()


def blocks_to_label_mask(
    shape_hw: Tuple[int, int],
    blocks: Sequence[InfoBlock],
    fill_bbox: bool = True,
) -> np.ndarray:
    """Rasterize blocks into an int32 label image (0=background, 1..=block id+1).

    If fill_bbox is True, fill the whole rectangle; else require block.mask.
    """
    label = np.zeros(shape_hw, dtype=np.int32)
    for block in blocks:
        class_id = block.id + 1
        if fill_bbox:
            label[block.y : block.y + block.h, block.x : block.x + block.w] = class_id
        elif block.mask is not None:
            label[block.mask > 0] = class_id
    return label
