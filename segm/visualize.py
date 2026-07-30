"""Visualization helpers for info-block segmentation debug.

Primary showcase style matches the boss demo UI:
semi-transparent multi-color region masks overlaid on the original drawing
(区域颜色), not green bounding boxes alone.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .extract import InfoBlock


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def make_region_palette(n: int, seed: int = 42) -> np.ndarray:
    """Distinct BGR colors for n regions (HSV spaced, similar to demo UI)."""
    n = max(int(n), 1)
    palette = np.zeros((n, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    # Shuffle hue order a bit so adjacent labels are less likely similar
    hues = np.linspace(0, 179, n, endpoint=False)
    rng.shuffle(hues)
    for i, hue in enumerate(hues):
        hsv = np.uint8([[[int(hue), 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        palette[i] = bgr
    return palette


def color_for_label(label_id: int, seed: int = 42) -> np.ndarray:
    """Stable BGR color for a label id (does not reshuffle when labels are added)."""
    lid = max(1, int(label_id))
    # Golden-angle hue walk + light seed jitter so nearby ids stay distinct.
    hue = int((lid * 47 + seed * 13) % 180)
    sat = 170 + (lid * 17 + seed) % 60
    val = 220 + (lid * 13) % 35
    hsv = np.uint8([[[hue, min(sat, 255), min(val, 255)]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]


def _bgr_lab_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Perceptual distance between two BGR colors (CIE Lab Euclidean)."""
    pair = np.stack([a, b]).astype(np.uint8).reshape(1, 2, 3)
    lab = cv2.cvtColor(pair, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(2, 3)
    d = lab[0] - lab[1]
    return float(np.sqrt(np.dot(d, d)))


def _expand_xyxy(
    x: int, y: int, w: int, h: int, margin: int
) -> Tuple[int, int, int, int]:
    return x - margin, y - margin, x + w + margin, y + h + margin


def _rects_overlap(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def _union_bboxes_by_key(
    blocks: Sequence[InfoBlock],
) -> dict:
    """Union axis-aligned bboxes for each color key (group_label or cc_label)."""
    out: dict = {}
    for block in blocks:
        if block.cc_label <= 0:
            continue
        key = int(block.group_label or block.cc_label)
        x0, y0 = int(block.x), int(block.y)
        x1, y1 = x0 + int(block.w), y0 + int(block.h)
        if key not in out:
            out[key] = [x0, y0, x1, y1]
        else:
            b = out[key]
            b[0] = min(b[0], x0)
            b[1] = min(b[1], y0)
            b[2] = max(b[2], x1)
            b[3] = max(b[3], y1)
    return {
        k: (v[0], v[1], max(1, v[2] - v[0]), max(1, v[3] - v[1]))
        for k, v in out.items()
    }


def _bboxes_from_label_map(
    label_map: np.ndarray, keys: Sequence[int]
) -> dict:
    """Cheap AABB per label id (one pass over nonzero pixels)."""
    key_set = {int(k) for k in keys}
    if not key_set:
        return {}
    ys, xs = np.nonzero(label_map)
    if xs.size == 0:
        return {}
    labs = label_map[ys, xs]
    out: dict = {}
    for key in key_set:
        sel = labs == key
        if not np.any(sel):
            continue
        kx = xs[sel]
        ky = ys[sel]
        x0, x1 = int(kx.min()), int(kx.max())
        y0, y1 = int(ky.min()), int(ky.max())
        out[key] = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    return out


def _neighbor_graph(
    keys: Sequence[int],
    bbox_by_key: dict,
    margin: int,
) -> dict:
    """Undirected adjacency: expanded bboxes overlap → neighbors."""
    neighbors = {int(k): [] for k in keys}
    key_list = [int(k) for k in keys]
    expanded = {}
    for k in key_list:
        box = bbox_by_key.get(k)
        if box is None:
            continue
        x, y, w, h = box
        expanded[k] = _expand_xyxy(x, y, w, h, margin)
    for i, ka in enumerate(key_list):
        ea = expanded.get(ka)
        if ea is None:
            continue
        for kb in key_list[i + 1 :]:
            eb = expanded.get(kb)
            if eb is None:
                continue
            if _rects_overlap(ea, eb):
                neighbors[ka].append(kb)
                neighbors[kb].append(ka)
    return neighbors


def _contrast_palette(n: int, seed: int = 42) -> List[np.ndarray]:
    """Fixed high-chroma BGR candidates (golden-angle hues + sat/val variants)."""
    n = max(int(n), 1)
    out: List[np.ndarray] = []
    # OpenCV hue is 0..179 for 0..360°; golden angle ≈ 137.508° → ≈ 68.75 units.
    step = 68.75
    for i in range(n):
        hue = int((i * step + seed * 7) % 180)
        sat = 175 + (i * 41 + seed) % 70
        val = 215 + (i * 23) % 40
        hsv = np.uint8([[[hue, min(sat, 255), min(val, 255)]]])
        out.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    return out


def assign_contrast_colors(
    color_keys: Sequence[int],
    bbox_by_key: dict,
    seed: int = 42,
    margin: int = 40,
    min_lab_dist: float = 45.0,
) -> dict:
    """Assign BGR colors so spatially nearby regions stay perceptually distinct.

    Prefers ``color_for_label`` for stability (split/merge keep old hues when
    possible); only recolors a key when its preferred color is too close to an
    already-assigned neighbor.
    """
    keys = sorted({int(k) for k in color_keys})
    if not keys:
        return {}

    neighbors = _neighbor_graph(keys, bbox_by_key, margin=max(0, int(margin)))
    palette = _contrast_palette(max(48, len(keys) * 3), seed=seed)
    assigned: dict = {}

    # Low ids first so existing regions keep preferred colors when new labels appear.
    for key in keys:
        preferred = color_for_label(key, seed=seed)
        neigh_cols = [assigned[n] for n in neighbors[key] if n in assigned]
        if not neigh_cols:
            assigned[key] = preferred
            continue
        pref_d = min(_bgr_lab_dist(preferred, c) for c in neigh_cols)
        if pref_d >= min_lab_dist:
            assigned[key] = preferred
            continue

        best = preferred
        best_d = pref_d
        for cand in palette:
            d = min(_bgr_lab_dist(cand, c) for c in neigh_cols)
            if d > best_d:
                best_d = d
                best = cand
        assigned[key] = best
    return assigned


def effective_kernel_size(gap_thres: int) -> int:
    """Return the actual centered kernel size used by extraction."""
    k = max(1, int(gap_thres))
    return k if k % 2 == 1 else k + 1


def draw_kernel_legend(
    img: np.ndarray,
    gap_thres: int,
    shape: str = "rect",
) -> np.ndarray:
    """Draw a readable kernel-size legend on the output image."""
    canvas = _ensure_bgr(img)
    h, w = canvas.shape[:2]
    k = effective_kernel_size(gap_thres)
    scale = max(0.6, min(h, w) / 1800.0)
    font_scale = 0.65 * scale
    thickness = max(1, int(round(2 * scale)))
    text = f"DILATE KERNEL: {k}x{k} px  |  GAP: {gap_thres} px  |  {shape.upper()}"
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    pad = max(8, int(round(10 * scale)))
    x0, y0 = pad, pad
    x1 = min(w - 1, x0 + tw + 2 * pad)
    y1 = min(h - 1, y0 + th + baseline + 2 * pad)
    layer = canvas.copy()
    cv2.rectangle(layer, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.addWeighted(layer, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (25, 25, 25), thickness)
    cv2.putText(
        canvas,
        text,
        (x0 + pad, y0 + pad + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )
    return canvas


def render_kernel_preview(gap_thres: int, shape: str = "rect") -> np.ndarray:
    """Render the exact morphology kernel as a visible grid image."""
    k = effective_kernel_size(gap_thres)
    morph = {
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }.get(shape.lower(), cv2.MORPH_RECT)
    kernel = cv2.getStructuringElement(morph, (k, k))
    cell = max(8, min(32, 480 // max(k, 1)))
    grid = np.full((k * cell, k * cell, 3), 245, dtype=np.uint8)
    for row in range(k):
        for col in range(k):
            color = (50, 180, 255) if kernel[row, col] else (255, 255, 255)
            p0 = (col * cell, row * cell)
            p1 = ((col + 1) * cell - 1, (row + 1) * cell - 1)
            cv2.rectangle(grid, p0, p1, color, -1)
            cv2.rectangle(grid, p0, p1, (80, 80, 80), 1)
    return grid


def draw_blocks(
    img: np.ndarray,
    blocks: Sequence[InfoBlock],
    color: tuple = (0, 180, 0),
    thickness: int = 2,
    draw_ids: bool = True,
) -> np.ndarray:
    """Draw block rectangles on a copy of img (debug aid)."""
    canvas = _ensure_bgr(img)
    for block in blocks:
        p1 = (block.x, block.y)
        p2 = (block.x + block.w, block.y + block.h)
        cv2.rectangle(canvas, p1, p2, color, thickness)
        if draw_ids:
            cv2.putText(
                canvas,
                str(block.id),
                (block.x, max(0, block.y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas


def colorize_label_map(label_map: np.ndarray, seed: int = 0) -> np.ndarray:
    """Map connected-component ids to a solid BGR color image (no alpha)."""
    rng = np.random.default_rng(seed)
    max_id = int(label_map.max())
    palette_size = min(max_id + 1, 1024)
    palette = rng.integers(40, 255, size=(palette_size, 3), dtype=np.uint8)
    palette[0] = 0
    if max_id + 1 <= palette_size:
        return palette[label_map]
    indexed = label_map.copy()
    nonzero = indexed > 0
    indexed[nonzero] = (indexed[nonzero] % (palette_size - 1)) + 1
    return palette[indexed]


def filtered_label_map(
    label_map: np.ndarray,
    blocks: Sequence[InfoBlock],
) -> np.ndarray:
    """Keep only CC labels that correspond to accepted info blocks."""
    keep = {int(b.cc_label) for b in blocks if b.cc_label > 0}
    if not keep:
        return np.zeros_like(label_map)
    # Vectorized keep via LUT
    max_id = int(label_map.max())
    lut = np.zeros(max_id + 1, dtype=np.int32)
    for lid in keep:
        if 0 < lid <= max_id:
            lut[lid] = lid
    return lut[label_map]


def draw_region_overlay(
    img: np.ndarray,
    label_map: np.ndarray,
    blocks: Optional[Sequence[InfoBlock]] = None,
    alpha: float = 0.45,
    draw_contours: bool = True,
    contour_thickness: int = 2,
    seed: int = 42,
    fill_mode: str = "dilated",
) -> np.ndarray:
    """Boss-demo style: semi-transparent colored region masks on the drawing.

    Parameters
    ----------
    img : original BGR/gray drawing
    label_map : dilated connected-component labels (from extract debug)
    blocks : if given, only paint accepted blocks (via cc_label)
    alpha : overlay strength in [0, 1]
    draw_contours : outline each region
    fill_mode :
        - ``"dilated"`` / ``"pixels"``: use the exact dilated CC boundary
        - ``"hull"``: fill convex hull of each region
        - ``"bbox"``: fill axis-aligned bbox of each block
    """
    base = _ensure_bgr(img)
    h, w = base.shape[:2]

    # Paint by stable group/cc label ids so split/merge does not reshuffle colors.
    if blocks is not None:
        labels = filtered_label_map(label_map, blocks)
        paint_pairs: List[Tuple[int, int]] = []  # (cc_label, color_key)
        for block in blocks:
            if block.cc_label <= 0:
                continue
            color_key = int(block.group_label or block.cc_label)
            paint_pairs.append((int(block.cc_label), color_key))
        # Deduplicate color keys for contour drawing.
        color_keys = sorted({ck for _, ck in paint_pairs})
        n_colors = len(color_keys)
        block_by_key = {
            int(b.group_label or b.cc_label): b
            for b in blocks
            if b.cc_label > 0
        }
    else:
        labels = label_map
        max_id = int(label_map.max()) if label_map.size else 0
        paint_pairs = [(i, i) for i in range(1, max_id + 1)]
        color_keys = [i for i in range(1, max_id + 1)]
        n_colors = len(color_keys)
        block_by_key = {}

    if n_colors <= 0:
        return base

    fill_mode = (fill_mode or "dilated").lower()
    # Neighbor-aware colors: nearby regions get higher contrast when hash collides.
    if blocks is not None:
        bbox_by_key = _union_bboxes_by_key(blocks)
    else:
        bbox_by_key = _bboxes_from_label_map(labels, color_keys)
    margin = max(24, int(0.02 * max(h, w)))
    key_to_color = assign_contrast_colors(
        color_keys,
        bbox_by_key,
        seed=seed,
        margin=margin,
        min_lab_dist=45.0,
    )
    color_img = np.zeros_like(base)
    region_mask = np.zeros((h, w), dtype=bool)
    paint_keys: Optional[np.ndarray] = None

    if fill_mode in ("dilated", "pixels"):
        # Remap cc_label → color_key, then palette LUT paint (O(pixels), one pass).
        max_lab = int(labels.max()) if labels.size else 0
        remap = np.zeros(max_lab + 1, dtype=np.int32)
        for cc_label, color_key in paint_pairs:
            if 0 < int(cc_label) <= max_lab:
                remap[int(cc_label)] = int(color_key)
        paint_keys = remap[labels]
        region_mask = paint_keys > 0
        if np.any(region_mask):
            max_key = int(paint_keys.max())
            palette = np.zeros((max_key + 1, 3), dtype=np.uint8)
            for ck, color in key_to_color.items():
                if 0 < int(ck) <= max_key:
                    palette[int(ck)] = color
            color_img = palette[paint_keys]
    else:
        for cc_label, color_key in paint_pairs:
            color = key_to_color[color_key]
            if fill_mode == "bbox" and color_key in block_by_key:
                b = block_by_key[color_key]
                x0, y0 = max(0, b.x), max(0, b.y)
                x1, y1 = min(w, b.x + b.w), min(h, b.y + b.h)
                color_img[y0:y1, x0:x1] = color
                region_mask[y0:y1, x0:x1] = True
                continue

            mask_u8 = (labels == cc_label).astype(np.uint8) * 255
            if not np.any(mask_u8):
                continue

            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            pts = np.concatenate(contours, axis=0)
            if len(pts) < 3:
                sel = mask_u8 > 0
                color_img[sel] = color
                region_mask[sel] = True
                continue
            hull = cv2.convexHull(pts)
            layer = np.zeros_like(base)
            cv2.fillConvexPoly(
                layer, hull, (int(color[0]), int(color[1]), int(color[2]))
            )
            sel = np.any(layer > 0, axis=2)
            color_img[sel] = color
            region_mask[sel] = True

    if not np.any(region_mask):
        return base

    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = base.astype(np.float32)
    color_f = color_img.astype(np.float32)
    overlay[region_mask] = (1.0 - alpha) * overlay[region_mask] + alpha * color_f[region_mask]
    overlay = overlay.astype(np.uint8)

    # Contour tracing is O(regions); skip when disabled or too many pieces.
    if draw_contours and n_colors <= 120:
        for color_key in color_keys:
            if paint_keys is not None:
                src = (paint_keys == int(color_key)).astype(np.uint8) * 255
            else:
                src = np.zeros((h, w), dtype=np.uint8)
                for cc_label, ck in paint_pairs:
                    if ck == color_key:
                        src[labels == cc_label] = 255
            if not np.any(src):
                continue
            # Prefer block bbox ROI so edit refresh stays fast on large pages.
            bb = bbox_by_key.get(int(color_key))
            if bb is not None:
                bx, by, bw, bh = bb
                pad = max(4, contour_thickness + 2)
                x0 = max(0, int(bx) - pad)
                y0 = max(0, int(by) - pad)
                x1 = min(w, int(bx) + int(bw) + pad)
                y1 = min(h, int(by) + int(bh) + pad)
                roi = src[y0:y1, x0:x1]
                contours, _ = cv2.findContours(
                    roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                # Shift contours back to full-image coords.
                contours = [
                    c + np.array([[[x0, y0]]], dtype=c.dtype) for c in contours
                ]
            else:
                contours, _ = cv2.findContours(
                    src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
            if not contours:
                continue
            color = tuple(int(c) for c in key_to_color[color_key])
            if fill_mode in ("dilated", "pixels"):
                cv2.drawContours(
                    overlay,
                    contours,
                    -1,
                    color,
                    contour_thickness,
                    cv2.LINE_AA,
                )
            else:
                pts = np.concatenate(contours, axis=0)
                if len(pts) < 3:
                    continue
                hull = cv2.convexHull(pts)
                cv2.polylines(
                    overlay,
                    [hull],
                    True,
                    color,
                    contour_thickness,
                    cv2.LINE_AA,
                )

    return overlay


def draw_region_overlay_from_debug(
    img: np.ndarray,
    blocks: Sequence[InfoBlock],
    debug: dict,
    alpha: float = 0.45,
    fill_mode: str = "dilated",
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper using extract_info_blocks debug dict."""
    return draw_region_overlay(
        img,
        debug["label_map"],
        blocks=blocks,
        alpha=alpha,
        fill_mode=fill_mode,
        **kwargs,
    )


def save_debug_bundle(
    output_dir: str,
    img: np.ndarray,
    blocks: List[InfoBlock],
    debug: dict,
    prefix: str = "info_blocks",
    alpha: float = 0.45,
    fill_mode: str = "dilated",
    show_kernel_legend: bool = False,
) -> dict:
    """Write intermediate images + boss-style region color overlay.

    Returns a dict of written file paths. Key result:
    ``08_region_overlay`` / ``region_overlay`` — translucent multi-color masks.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    def _write(name: str, array: np.ndarray) -> str:
        path = os.path.join(output_dir, f"{prefix}_{name}.png")
        cv2.imwrite(path, array)
        paths[name] = path
        return path

    _write("00_input", _ensure_bgr(img))
    _write("01_gray", debug["gray"])
    _write("02_binary", debug["binary"])
    _write("03_border_cleared", debug["binary_border_cleared"])
    _write("04_lines_removed", debug["lines"])
    _write("05_binary_for_dilate", debug["binary_nolines"])
    _write("06_dilated", debug["dilated"])
    _write("07_cc_colors", colorize_label_map(debug["label_map"]))
    _write("08_blocks", draw_blocks(img, blocks))

    overlay = draw_region_overlay_from_debug(
        img, blocks, debug, alpha=alpha, fill_mode=fill_mode
    )
    cfg = debug.get("config")
    effective_gap = int(
        debug.get("effective_gap_thres", cfg.gap_thres if cfg is not None else 1)
    )
    if cfg is not None and show_kernel_legend:
        overlay = draw_kernel_legend(
            overlay, effective_gap, cfg.dilate_shape
        )
    _write("09_region_overlay", overlay)
    alias = os.path.join(output_dir, "region_overlay.png")
    cv2.imwrite(alias, overlay)
    paths["region_overlay"] = alias

    if cfg is not None:
        _write(
            "10_dilation_kernel",
            render_kernel_preview(effective_gap, cfg.dilate_shape),
        )
    meta_path = os.path.join(output_dir, f"{prefix}_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"num_blocks={len(blocks)}\n")
        f.write(f"overlay_alpha={alpha}\n")
        f.write(f"fill_mode={fill_mode}\n")
        f.write(f"border_margin_px={debug.get('border_margin', 0)}\n")
        if cfg is not None:
            k = effective_kernel_size(effective_gap)
            f.write(f"requested_gap={cfg.gap_thres}\n")
            f.write(f"effective_gap={effective_gap}\n")
            f.write(f"effective_kernel={k}x{k}\n")
        if cfg is not None:
            for key, value in vars(cfg).items():
                f.write(f"{key}={value}\n")
        f.write("\nblocks (id,cc_label,group_label,x,y,w,h,area):\n")
        for b in blocks:
            f.write(
                f"{b.id},{b.cc_label},{b.group_label},"
                f"{b.x},{b.y},{b.w},{b.h},{b.area}\n"
            )
    paths["meta"] = meta_path
    return paths
