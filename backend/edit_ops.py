"""Manual split / merge edits on a page label_map + blocks."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from segm.extract import InfoBlock


Point = Tuple[float, float]


def _stroke_mask(
    shape: Tuple[int, int],
    points: Sequence[Point],
    width: int,
) -> np.ndarray:
    """Rasterize a polyline stroke into a binary mask."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 2:
        return mask
    pts = np.array(
        [[int(round(x)), int(round(y))] for x, y in points],
        dtype=np.int32,
    )
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    thickness = max(1, int(width))
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)
    if thickness > 1:
        k = thickness if thickness % 2 == 1 else thickness + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)
    return mask


def _labels_touched(
    label_map: np.ndarray,
    stroke: np.ndarray,
) -> List[int]:
    vals = label_map[stroke > 0]
    vals = vals[vals > 0]
    if vals.size == 0:
        return []
    return [int(v) for v in np.unique(vals)]


def _target_roi(
    region_bool: np.ndarray,
    points: Sequence[Point],
    shape: Tuple[int, int],
    pad: int,
) -> Tuple[int, int, int, int]:
    """BBox of target region ∪ stroke, padded and clipped to the image."""
    h, w = shape
    x, y, bw, bh = cv2.boundingRect(region_bool.astype(np.uint8))
    if bw <= 0 or bh <= 0:
        return 0, 0, w, h
    x0, y0, x1, y1 = int(x), int(y), int(x + bw), int(y + bh)
    if points:
        pxs = [float(p[0]) for p in points]
        pys = [float(p[1]) for p in points]
        x0 = min(x0, int(np.floor(min(pxs))))
        y0 = min(y0, int(np.floor(min(pys))))
        x1 = max(x1, int(np.ceil(max(pxs))) + 1)
        y1 = max(y1, int(np.ceil(max(pys))) + 1)
    x0 = int(np.clip(x0 - pad, 0, w - 1))
    y0 = int(np.clip(y0 - pad, 0, h - 1))
    x1 = int(np.clip(x1 + pad, x0 + 1, w))
    y1 = int(np.clip(y1 + pad, y0 + 1, h))
    return x0, y0, x1, y1


def _absorb_tiny_fragments(
    label_map: np.ndarray,
    region_mask: np.ndarray,
    keep_labels: Sequence[int],
) -> np.ndarray:
    """Merge leftover scraps of the edited region into the nearest kept label."""
    # Caller already owns a writable copy; avoid a second full-image clone.
    out = label_map
    keep = [int(x) for x in keep_labels]
    if len(keep) < 1:
        return out

    rx, ry, rbw, rbh = cv2.boundingRect(region_mask.astype(np.uint8))
    if rbw <= 0 or rbh <= 0:
        return out
    # Pad so fragment dilate can see keep labels just outside the region bbox.
    pad = 5
    h, w = out.shape[:2]
    x0 = max(0, int(rx) - pad)
    y0 = max(0, int(ry) - pad)
    x1 = min(w, int(rx + rbw) + pad)
    y1 = min(h, int(ry + rbh) + pad)

    roi_out = out[y0:y1, x0:x1]
    roi_region = region_mask[y0:y1, x0:x1]
    leftover = roi_region & ~np.isin(roi_out, keep)
    if not np.any(leftover):
        return out

    n, cc, stats, cents = cv2.connectedComponentsWithStats(
        leftover.astype(np.uint8) * 255, connectivity=8
    )
    keep_cents: dict = {}
    for lab in keep:
        m = roi_out == lab
        if not np.any(m):
            continue
        ys, xs = np.where(m)
        keep_cents[lab] = (float(ys.mean()), float(xs.mean()))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for i in range(1, n):
        frag = cc == i
        ring = cv2.dilate(frag.astype(np.uint8) * 255, k) > 0
        ring = ring & ~frag
        best_lab = keep[0]
        best_hit = -1
        for lab in keep:
            hit = int(((roi_out == lab) & ring).sum())
            if hit > best_hit:
                best_hit = hit
                best_lab = lab
        if best_hit <= 0:
            cy = float(cents[i][1])
            cx = float(cents[i][0])
            best_d = 1e18
            for lab, (ly, lx) in keep_cents.items():
                d = (ly - cy) ** 2 + (lx - cx) ** 2
                if d < best_d:
                    best_d = d
                    best_lab = lab
        roi_out[frag] = best_lab
    return out


def _rebuild_block_from_mask(
    binary: np.ndarray,
    label_map: np.ndarray,
    label_id: int,
    block_id: int,
    group_label: int = 0,
) -> Optional[InfoBlock]:
    """Build an InfoBlock from ink under a label."""
    region_u8 = (label_map == int(label_id)).astype(np.uint8)
    x0, y0, bw, bh = cv2.boundingRect(region_u8)
    if bw <= 0 or bh <= 0:
        return None
    x1, y1 = x0 + bw, y0 + bh
    ink = binary[y0:y1, x0:x1] > 0
    area = int(ink.sum()) if ink.size else int(region_u8.sum())
    if area <= 0:
        area = int(region_u8.sum())
    return InfoBlock(
        id=block_id,
        x=x0,
        y=y0,
        w=bw,
        h=bh,
        area=area,
        cc_label=int(label_id),
        group_label=int(group_label or label_id),
    )


def _renumber_blocks(blocks: List[InfoBlock]) -> List[InfoBlock]:
    """Only renumber display ids; keep cc_label/group_label stable for colors."""
    out: List[InfoBlock] = []
    for i, b in enumerate(sorted(blocks, key=lambda t: (t.y, t.x, t.id)), start=1):
        out.append(
            InfoBlock(
                id=i,
                x=b.x,
                y=b.y,
                w=b.w,
                h=b.h,
                area=b.area,
                cc_label=b.cc_label,
                group_label=b.group_label or b.cc_label,
            )
        )
    return out


def _stroke_side_sign(
    cx: float,
    cy: float,
    points: Sequence[Point],
) -> float:
    """Signed side of point relative to the stroke (first→last as axis)."""
    if len(points) < 2:
        return 0.0
    x0, y0 = float(points[0][0]), float(points[0][1])
    x1, y1 = float(points[-1][0]), float(points[-1][1])
    dx, dy = x1 - x0, y1 - y0
    if abs(dx) + abs(dy) < 1e-6:
        for i in range(len(points) - 1):
            dx = float(points[i + 1][0]) - float(points[i][0])
            dy = float(points[i + 1][1]) - float(points[i][1])
            if abs(dx) + abs(dy) >= 1e-6:
                x0, y0 = float(points[i][0]), float(points[i][1])
                break
    return (cx - x0) * dy - (cy - y0) * dx


def _partition_by_stroke_sides(
    cc: np.ndarray,
    fg_ids: Sequence[int],
    points: Sequence[Point],
    binary_roi: np.ndarray,
    region_roi: np.ndarray,
    stats: Optional[np.ndarray] = None,
    centroids: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[int], List[int], str]:
    """Group all CCs onto two sides of the stroke; return (side_a, side_b, ink, fail)."""
    if len(fg_ids) < 2:
        return [], [], [], "not_disconnected"

    side_pos: List[int] = []
    side_neg: List[int] = []
    ink = (binary_roi > 0) & region_roi
    # One-pass ink counts per CC id (avoids repeated np.isin scans).
    ink_ids = cc[ink]
    if ink_ids.size:
        ink_bc = np.bincount(ink_ids.ravel())
    else:
        ink_bc = np.zeros(1, dtype=np.int64)

    for fid in fg_ids:
        fid = int(fid)
        if centroids is not None and fid < len(centroids):
            cx, cy = float(centroids[fid][0]), float(centroids[fid][1])
        else:
            ys, xs = np.where(cc == fid)
            if len(xs) == 0:
                continue
            cy, cx = float(ys.mean()), float(xs.mean())
        if _stroke_side_sign(cx, cy, points) >= 0:
            side_pos.append(fid)
        else:
            side_neg.append(fid)

    if not side_pos or not side_neg:
        if stats is not None:
            areas_map = {
                int(fid): int(stats[int(fid), cv2.CC_STAT_AREA]) for fid in fg_ids
            }
        else:
            areas_map = {int(fid): int((cc == fid).sum()) for fid in fg_ids}
        ordered = sorted(fg_ids, key=lambda i: areas_map[int(i)], reverse=True)
        side_pos = [int(ordered[0])]
        side_neg = [int(i) for i in ordered[1:]]
        if not side_neg:
            return [], [], [], "not_disconnected"

    def _ink_of(ids: Sequence[int]) -> int:
        total = 0
        for i in ids:
            ii = int(i)
            if 0 <= ii < len(ink_bc):
                total += int(ink_bc[ii])
        return total

    def _area_of(ids: Sequence[int]) -> int:
        if not ids:
            return 0
        if stats is not None:
            return int(sum(int(stats[int(i), cv2.CC_STAT_AREA]) for i in ids))
        return int(np.isin(cc, list(ids)).sum())

    ink_counts = [_ink_of(side_pos), _ink_of(side_neg)]
    areas = [_area_of(side_pos), _area_of(side_neg)]
    if min(areas) < 20:
        return [], [], ink_counts, "empty_side"
    if max(ink_counts) > 0 and min(ink_counts) == 0 and min(areas) < 80:
        return [], [], ink_counts, "empty_side"
    return side_pos, side_neg, ink_counts, ""


def _peel_minority_near_stroke(
    region_roi: np.ndarray,
    binary_roi: np.ndarray,
    local_pts: Sequence[Point],
    cut_w: int,
) -> Tuple[Optional[np.ndarray], dict, str]:
    """Peel stroke-selected minority ink (+ nearer halo) out of an engulfing region.

    Handles ``M 1:2`` glued between twin views: a hard bisect would split the
    views, but the user wants the small annotation as the new-colored piece.
    Returns a bool mask for the small piece inside the ROI, or fail.
    """
    meta: dict = {"mode": "stroke_peel", "selected_islands": [], "peel_ink_frac": 0.0}
    region_bool = region_roi > 0
    ink = (binary_roi > 0) & region_bool
    if not np.any(ink):
        return None, meta, "no_ink"

    total_ink = int(ink.sum())
    rh, rw = region_roi.shape
    probe_w = max(14, int(cut_w) * 3)
    corridor = _stroke_mask((rh, rw), local_pts, probe_w)
    corridor = cv2.bitwise_and(corridor, (region_bool.astype(np.uint8) * 255))
    if int((corridor > 0).sum()) < 3:
        return None, meta, "no_stroke"

    ink_u8 = ink.astype(np.uint8) * 255
    n_ink, ink_cc, ink_stats, ink_cents = cv2.connectedComponentsWithStats(
        ink_u8, connectivity=8
    )
    if n_ink <= 2:
        # Single solid ink body: peel by proximity to stroke only.
        dist = cv2.distanceTransform(
            (corridor == 0).astype(np.uint8), cv2.DIST_L2, 3
        )
        near = (dist <= float(max(18, 4 * cut_w))) & ink
        peel_ink = int(near.sum())
        frac = peel_ink / float(max(total_ink, 1))
        meta["peel_ink_frac"] = float(frac)
        meta["mode"] = "stroke_peel_proximity"
        if peel_ink < 12 or frac <= 0.0 or frac > 0.28:
            return None, meta, "not_minority"
        selected_ink = near
    else:
        # Grow corridor until it touches ink, collect hit islands.
        grow = corridor.copy()
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for _ in range(max(60, 14 * cut_w)):
            if np.any(grow[ink]):
                break
            nxt = cv2.dilate(grow, k3)
            nxt[~region_bool] = 0
            if int((nxt > 0).sum()) == int((grow > 0).sum()):
                break
            grow = nxt
        hit = np.unique(ink_cc[grow > 0])
        hit_ids = sorted(int(v) for v in hit if int(v) > 0)
        if not hit_ids:
            return None, meta, "no_ink_near_stroke"

        areas = {i: int(ink_stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_ink)}
        soft_cut = 0.35 * total_ink
        selected = {i for i in hit_ids if areas[i] < soft_cut}
        if not selected:
            return None, meta, "hit_main_body"

        # Absorb nearby small glyph islands (``M`` ``1`` ``:`` ``2``).
        # OpenCV centroids are (x, y); store as (cy, cx) for distance math.
        cents = {
            i: (float(ink_cents[i][1]), float(ink_cents[i][0]))
            for i in range(1, n_ink)
            if areas[i] > 0
        }
        link = float(max(48, 10 * cut_w))
        sel_areas = [areas[i] for i in selected]
        if sel_areas:
            med = float(np.median(sel_areas))
            link = max(link, 2.5 * np.sqrt(max(med, 1.0)))
        minority_cut = 0.12 * total_ink
        changed = True
        while changed:
            changed = False
            for i in range(1, n_ink):
                if i in selected or areas[i] >= minority_cut or i not in cents:
                    continue
                cy, cx = cents[i]
                for j in selected:
                    if j not in cents:
                        continue
                    jy, jx = cents[j]
                    if (cy - jy) ** 2 + (cx - jx) ** 2 <= link * link:
                        selected.add(i)
                        changed = True
                        break

        selected_ids = sorted(selected)
        peel_ink = sum(areas[i] for i in selected_ids)
        frac = peel_ink / float(max(total_ink, 1))
        meta["selected_islands"] = selected_ids
        meta["peel_ink_frac"] = float(frac)
        if frac <= 0.0 or frac > 0.35:
            return None, meta, "not_minority"
        selected_ink = np.isin(ink_cc, selected_ids)

    # Claim halo nearer to selected ink than to remaining ink.
    other_ink = ink & ~selected_ink
    if not np.any(other_ink):
        return None, meta, "no_remainder"
    inv_sel = (~selected_ink).astype(np.uint8) * 255
    inv_oth = (~other_ink).astype(np.uint8) * 255
    d_sel = cv2.distanceTransform(inv_sel, cv2.DIST_L2, 3)
    d_oth = cv2.distanceTransform(inv_oth, cv2.DIST_L2, 3)
    halo = region_bool & ~ink
    claim = halo & (d_sel <= d_oth) & (d_sel <= float(max(28, 6 * cut_w)))
    small = selected_ink | claim
    if int(small.sum()) < 20:
        return None, meta, "too_small"
    return small, meta, ""


def split_by_stroke(
    label_map: np.ndarray,
    blocks: Sequence[InfoBlock],
    binary: np.ndarray,
    points: Sequence[Point],
    stroke_width: int = 5,
) -> Tuple[np.ndarray, List[InfoBlock], dict]:
    """Split a glued component along the user stroke.

    1) Hard-cut + group CCs by stroke side (图一图二 style glue neck).
    2) If that would bisect two large siblings, or hard-cut fails: peel the
       minority annotation near the stroke (``M 1:2``) as the new-colored piece.

    Larger piece keeps the original label/color; smaller gets a new label.
    """
    meta = {
        "ok": False,
        "reason": "",
        "split_label": 0,
        "new_labels": [],
        "snap": {"mode": "stroke_cut"},
    }
    if label_map is None or len(points) < 2:
        meta["reason"] = "need at least 2 stroke points"
        return label_map, list(blocks), meta

    h, w = label_map.shape[:2]
    if binary is None or binary.shape[:2] != (h, w):
        meta["reason"] = "missing binary mask for split"
        return label_map, list(blocks), meta

    cut_w = max(3, int(stroke_width))
    probe = _stroke_mask((h, w), points, max(cut_w, 8))
    probe_vals = label_map[probe > 0]
    probe_vals = probe_vals[probe_vals > 0]
    if probe_vals.size == 0:
        meta["reason"] = "stroke did not hit any component"
        return label_map, list(blocks), meta

    counts_bc = np.bincount(probe_vals.ravel())
    target = int(np.argmax(counts_bc))
    meta["split_label"] = target

    region_bool = label_map == target
    if not np.any(region_bool):
        meta["reason"] = "empty target component"
        return label_map, list(blocks), meta

    pad = max(32, 4 * cut_w)
    x0, y0, x1, y1 = _target_roi(region_bool, points, (h, w), pad=pad)
    meta["snap"]["roi"] = [x0, y0, x1, y1]

    region_roi = region_bool[y0:y1, x0:x1]
    binary_roi = binary[y0:y1, x0:x1]
    rh, rw = region_roi.shape
    local_pts = [(float(px) - x0, float(py) - y0) for px, py in points]
    total_ink = int(((binary_roi > 0) & region_roi).sum())

    corridor = _stroke_mask((rh, rw), local_pts, cut_w)
    corridor = cv2.bitwise_and(corridor, (region_roi.astype(np.uint8) * 255))

    def _try_cut(
        cut_mask: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], List[int], List[int], List[int], str]:
        if int((cut_mask > 0).sum()) < 3:
            return None, [], [], [], "cut_too_weak"
        work = (region_roi.astype(np.uint8) * 255).copy()
        work[cut_mask > 0] = 0
        n, cc, stats, cents = cv2.connectedComponentsWithStats(work, connectivity=8)
        fg = [i for i in range(1, n) if int(stats[i, cv2.CC_STAT_AREA]) > 0]
        if len(fg) < 2:
            return None, [], [], [], "not_disconnected"
        side_a, side_b, ink_counts, fail = _partition_by_stroke_sides(
            cc,
            fg,
            local_pts,
            binary_roi,
            region_roi,
            stats=stats,
            centroids=cents,
        )
        if fail:
            return None, [], [], ink_counts, fail
        return cc, side_a, side_b, ink_counts, ""

    cc, side_a, side_b, ink_counts, fail = _try_cut(corridor)
    mode = "stroke_cut"
    if fail:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cut_w + 6, cut_w + 6))
        fat = cv2.dilate(corridor, k)
        fat = cv2.bitwise_and(fat, (region_roi.astype(np.uint8) * 255))
        cc, side_a, side_b, ink_counts, fail = _try_cut(fat)
        mode = "stroke_cut_wide"

    # Lazy peel: only run when hard-cut fails, or when the cut looks balanced
    # (twin large views) and we may prefer peeling a small annotation instead.
    balanced = False
    if (
        not fail
        and ink_counts
        and len(ink_counts) >= 2
        and total_ink > 0
    ):
        a, b = int(ink_counts[0]), int(ink_counts[1])
        balanced = min(a, b) > 0.18 * total_ink and max(a, b) < 0.82 * total_ink

    peel_mask: Optional[np.ndarray] = None
    peel_meta: dict = {}
    peel_fail = ""
    peel_ok = False
    if fail or balanced:
        peel_mask, peel_meta, peel_fail = _peel_minority_near_stroke(
            region_roi, binary_roi, local_pts, cut_w
        )
        peel_ok = peel_mask is not None and not peel_fail

    use_peel = False
    if peel_ok and peel_mask is not None:
        if fail:
            use_peel = True
        elif balanced and peel_meta.get("selected_islands"):
            if float(peel_meta.get("peel_ink_frac") or 1.0) <= 0.20:
                use_peel = True

    keep_mask: Optional[np.ndarray] = None
    small_mask: Optional[np.ndarray] = None

    if use_peel and peel_mask is not None:
        small_mask = peel_mask
        keep_mask = region_roi & ~small_mask
        mode = str(peel_meta.get("mode") or "stroke_peel")
        meta["snap"] = {**meta["snap"], **peel_meta}
        ink_counts = [
            int(((keep_mask) & (binary_roi > 0)).sum()),
            int(((small_mask) & (binary_roi > 0)).sum()),
        ]
    elif not fail and cc is not None:
        max_cc = int(cc.max()) if cc.size else 0
        area_bc = np.bincount(cc.ravel(), minlength=max_cc + 1)
        area_a = int(sum(int(area_bc[i]) for i in side_a)) if side_a else 0
        area_b = int(sum(int(area_bc[i]) for i in side_b)) if side_b else 0
        if area_a >= area_b:
            keep_ids, small_ids = side_a, side_b
        else:
            keep_ids, small_ids = side_b, side_a
            if len(ink_counts) == 2:
                ink_counts = [ink_counts[1], ink_counts[0]]
        lut = np.zeros(max_cc + 1, dtype=np.uint8)
        if keep_ids:
            lut[np.asarray(keep_ids, dtype=np.int32)] = 1
        keep_mask = lut[cc].astype(bool) if keep_ids else np.zeros_like(cc, dtype=bool)
        lut[:] = 0
        if small_ids:
            lut[np.asarray(small_ids, dtype=np.int32)] = 1
        small_mask = lut[cc].astype(bool) if small_ids else np.zeros_like(cc, dtype=bool)
        meta["snap"]["side_counts"] = [len(keep_ids), len(small_ids)]
    elif peel_ok and peel_mask is not None:
        small_mask = peel_mask
        keep_mask = region_roi & ~small_mask
        mode = str(peel_meta.get("mode") or "stroke_peel")
        meta["snap"] = {**meta["snap"], **peel_meta}
        ink_counts = [
            int(((keep_mask) & (binary_roi > 0)).sum()),
            int(((small_mask) & (binary_roi > 0)).sum()),
        ]
    else:
        meta["reason"] = "未能断开（请把线画穿黏连处，或画在要拆出的文字上）"
        meta["snap"]["fail"] = fail or peel_fail
        meta["snap"]["ink_counts"] = ink_counts
        meta["snap"]["peel"] = peel_meta
        return label_map, list(blocks), meta

    assert keep_mask is not None and small_mask is not None
    new_map = label_map.astype(np.int32, copy=True)
    next_id = int(new_map.max()) + 1
    new_map[y0:y1, x0:x1][region_roi] = 0
    lab_keep = target
    lab_small = next_id
    new_map[y0:y1, x0:x1][keep_mask] = lab_keep
    new_map[y0:y1, x0:x1][small_mask] = lab_small
    new_labels = [lab_keep, lab_small]

    new_map = _absorb_tiny_fragments(new_map, region_bool, new_labels)

    kept = [b for b in blocks if int(b.cc_label) != target]
    max_bid = max((b.id for b in kept), default=0)
    for lab in new_labels:
        max_bid += 1
        nb = _rebuild_block_from_mask(binary, new_map, lab, max_bid, group_label=lab)
        if nb is not None:
            kept.append(nb)

    kept = _renumber_blocks(kept)
    meta["ok"] = True
    meta["new_labels"] = new_labels
    meta["snap"]["mode"] = mode
    meta["snap"]["ink_counts"] = ink_counts
    roi_map = new_map[y0:y1, x0:x1]
    meta["snap"]["areas"] = [
        int((roi_map == lab_keep).sum()),
        int((roi_map == lab_small).sum()),
    ]
    if mode.startswith("stroke_peel"):
        meta["reason"] = f"已剥出笔画处小块并换新色（{mode}）"
    else:
        meta["reason"] = f"已沿笔画拆成 2 块（{mode}）"
    return new_map, kept, meta


def delete_by_stroke(
    label_map: np.ndarray,
    blocks: Sequence[InfoBlock],
    points: Sequence[Point],
    hit_radius: int = 8,
) -> Tuple[np.ndarray, List[InfoBlock], dict]:
    """Remove components touched by a click/stroke."""
    meta = {"ok": False, "reason": "", "deleted_labels": [], "deleted_count": 0}
    if label_map is None or not points:
        meta["reason"] = "need a click or stroke on a component"
        return label_map, list(blocks), meta

    h, w = label_map.shape[:2]
    if len(points) == 1:
        stroke = np.zeros((h, w), dtype=np.uint8)
        x, y = int(round(points[0][0])), int(round(points[0][1]))
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        r = max(4, int(hit_radius))
        cv2.circle(stroke, (x, y), r, 255, -1)
    else:
        stroke = _stroke_mask((h, w), points, max(hit_radius, 6))

    touched = _labels_touched(label_map, stroke)
    if not touched:
        meta["reason"] = "未点中任何组件"
        return label_map, list(blocks), meta

    drop = set(int(x) for x in touched)
    new_map = label_map.astype(np.int32, copy=True)
    for lab in drop:
        new_map[new_map == lab] = 0

    kept = [b for b in blocks if int(b.cc_label) not in drop]
    for b in kept:
        if int(b.group_label) in drop:
            b.group_label = int(b.cc_label)

    kept = _renumber_blocks(kept)
    meta["ok"] = True
    meta["deleted_labels"] = sorted(drop)
    meta["deleted_count"] = len(drop)
    meta["reason"] = f"已删除 {len(drop)} 个组件"
    return new_map, kept, meta


def merge_by_stroke(
    label_map: np.ndarray,
    blocks: Sequence[InfoBlock],
    binary: np.ndarray,
    points: Sequence[Point],
    stroke_width: int = 6,
    seal_gap: int = 3,
) -> Tuple[np.ndarray, List[InfoBlock], dict]:
    """Merge components that a user stroke touches."""
    meta = {"ok": False, "reason": "", "merged_labels": [], "keep_label": 0}
    if label_map is None or len(points) < 2:
        meta["reason"] = "need at least 2 stroke points"
        return label_map, list(blocks), meta

    h, w = label_map.shape[:2]
    stroke = _stroke_mask((h, w), points, stroke_width)
    touched = _labels_touched(label_map, stroke)
    if len(touched) < 2:
        meta["reason"] = "stroke must touch at least two components"
        return label_map, list(blocks), meta

    keep = min(touched)
    others = [lab for lab in touched if lab != keep]
    meta["merged_labels"] = touched
    meta["keep_label"] = int(keep)

    new_map = label_map.astype(np.int32, copy=True)
    for lab in others:
        new_map[new_map == int(lab)] = int(keep)

    if seal_gap > 0:
        seed = (new_map == int(keep)).astype(np.uint8) * 255
        k = seal_gap if seal_gap % 2 == 1 else seal_gap + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        sealed = cv2.dilate(seed, kernel)
        paint = (sealed > 0) & ((new_map == 0) | (new_map == int(keep)))
        new_map[paint] = int(keep)

    drop = set(others)
    kept_blocks = [b for b in blocks if int(b.cc_label) not in drop]
    kept_blocks = [b for b in kept_blocks if int(b.cc_label) != int(keep)]
    max_bid = max((b.id for b in kept_blocks), default=0) + 1
    nb = _rebuild_block_from_mask(binary, new_map, keep, max_bid, group_label=keep)
    if nb is not None:
        kept_blocks.append(nb)

    for b in kept_blocks:
        if int(b.group_label) in drop:
            b.group_label = int(keep)

    kept_blocks = _renumber_blocks(kept_blocks)
    meta["ok"] = True
    meta["reason"] = f"merged {len(touched)} components into label {keep}"
    return new_map, kept_blocks, meta
