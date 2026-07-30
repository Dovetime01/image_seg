"""Export rectangular masked part crops from info-block segmentation."""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .extract import InfoBlock


def _group_blocks(blocks: Sequence[InfoBlock]) -> Dict[int, List[InfoBlock]]:
    groups: Dict[int, List[InfoBlock]] = {}
    for block in blocks:
        gid = int(block.group_label or block.cc_label)
        groups.setdefault(gid, []).append(block)
    return groups


def _union_bbox(
    members: Sequence[InfoBlock],
    image_shape: Tuple[int, int],
    pad: int,
) -> Tuple[int, int, int, int]:
    h, w = image_shape
    x0 = min(b.x for b in members)
    y0 = min(b.y for b in members)
    x1 = max(b.x + b.w for b in members)
    y1 = max(b.y + b.h for b in members)
    pad = max(0, int(pad))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return x0, y0, x1, y1


def export_part_crops(
    img: np.ndarray,
    blocks: Sequence[InfoBlock],
    label_map: np.ndarray,
    output_dir: str,
    pad: int = 0,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    min_bbox_area: int = 12000,
    min_mask_area: int = 2500,
) -> List[dict]:
    """Save one rectangular PNG per part group with foreign content masked out.

    The crop is the axis-aligned bbox of the group (plus optional pad).
    Pixels that do not belong to this group's dilated CC labels are filled with
    ``bg_color`` so other parts never appear in the export.
    Tiny border ticks / noise groups are skipped by ``min_bbox_area`` /
    ``min_mask_area``.
    """
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(output_dir):
        if name.endswith(".png") or name.endswith(".txt"):
            path = os.path.join(output_dir, name)
            if os.path.isfile(path):
                os.remove(path)

    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img

    h, w = img_bgr.shape[:2]
    groups = _group_blocks(blocks)
    if not groups:
        return []

    max_label = int(label_map.max()) if label_map.size else 0
    manifest: List[dict] = []
    sorted_gids = sorted(groups.keys())
    part_idx = 0

    for gid in sorted_gids:
        members = groups[gid]
        x0, y0, x1, y1 = _union_bbox(members, (h, w), pad=pad)
        if x1 <= x0 or y1 <= y0:
            continue
        if (x1 - x0) * (y1 - y0) < int(min_bbox_area):
            continue

        keep = np.zeros(max_label + 1, dtype=bool)
        for block in members:
            cc = int(block.cc_label)
            if 0 < cc <= max_label:
                keep[cc] = True

        labels_roi = label_map[y0:y1, x0:x1]
        mask = keep[labels_roi]
        if int(mask.sum()) < int(min_mask_area):
            continue

        crop = img_bgr[y0:y1, x0:x1]
        part = np.empty_like(crop)
        part[:] = bg_color
        part[mask] = crop[mask]

        filename = f"part_{part_idx:03d}.png"
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, part)

        record = {
            "part_idx": part_idx,
            "group_label": gid,
            "path": out_path,
            "x": x0,
            "y": y0,
            "w": x1 - x0,
            "h": y1 - y0,
            "member_count": len(members),
            "area_mask": int(mask.sum()),
        }
        manifest.append(record)
        part_idx += 1

    manifest_path = os.path.join(output_dir, "parts_manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("part_idx,group_label,x,y,w,h,member_count,area_mask,path\n")
        for item in manifest:
            f.write(
                f"{item['part_idx']},{item['group_label']},"
                f"{item['x']},{item['y']},{item['w']},{item['h']},"
                f"{item['member_count']},{item['area_mask']},{item['path']}\n"
            )
    return manifest
