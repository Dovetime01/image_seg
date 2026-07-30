"""Smoke tests for stroke-cut split (ROI hard cut along user stroke)."""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.edit_ops import split_by_stroke
from segm.extract import InfoBlock


def _block(lab: int, region: np.ndarray, bid: int = 1) -> InfoBlock:
    ys, xs = np.where(region)
    return InfoBlock(
        id=bid,
        x=int(xs.min()),
        y=int(ys.min()),
        w=int(xs.max() - xs.min() + 1),
        h=int(ys.max() - ys.min() + 1),
        area=int(region.sum()),
        cc_label=lab,
        group_label=lab,
    )


def _assert_stroke_mode(meta: dict) -> None:
    mode = str(meta.get("snap", {}).get("mode", ""))
    assert mode.startswith("stroke_cut"), mode


def test_two_rects_glued() -> None:
    h, w = 120, 200
    binary = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(binary, (20, 40), (70, 80), 255, -1)
    cv2.rectangle(binary, (100, 40), (150, 80), 255, -1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dilated = cv2.dilate(binary, k)
    label_map = (dilated > 0).astype(np.int32)
    blocks = [_block(1, label_map == 1)]
    new_map, _, meta = split_by_stroke(
        label_map, blocks, binary, [(85, 35), (85, 85)], stroke_width=5
    )
    assert meta["ok"], meta
    _assert_stroke_mode(meta)
    assert len(meta["new_labels"]) == 2
    # Larger piece keeps original label; smaller gets a new id.
    keep, small = meta["new_labels"]
    assert keep == 1 and small != 1
    assert int(new_map[60, 45]) == 1 or int(new_map[60, 125]) == 1
    assert int(new_map[60, 45]) != int(new_map[60, 125])


def test_upper_gap_distant_circle_untouched() -> None:
    """Upper text gap cut must not carve the distant circle into its own piece."""
    h, w = 320, 300
    binary = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(binary, (40, 30), (110, 70), 255, -1)
    cv2.rectangle(binary, (130, 30), (200, 70), 255, -1)
    cv2.circle(binary, (230, 250), 40, 255, 3)
    cv2.ellipse(binary, (230, 250), (60, 60), 0, 200, 340, 255, 2)
    cv2.line(binary, (165, 70), (230, 210), 255, 2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dilated = cv2.dilate(binary, k)
    _, cc = cv2.connectedComponents(dilated)
    ring_y, ring_x = 250, 270
    main = int(cc[50, 70])
    assert main > 0
    assert int(cc[50, 160]) == main and int(cc[ring_y, ring_x]) == main
    assert int(binary[50, 120]) == 0 and int(dilated[50, 120]) == 255

    label_map = (cc == main).astype(np.int32)
    blocks = [_block(1, label_map == 1)]
    new_map, _, meta = split_by_stroke(
        label_map, blocks, binary, [(120, 20), (120, 85)], stroke_width=5
    )
    assert meta["ok"], meta
    _assert_stroke_mode(meta)

    ul = int(new_map[50, 70])
    ur = int(new_map[50, 160])
    assert ul != ur

    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (230, 250), 40, 255, 3)
    circle_labs = {int(v) for v in new_map[circle_mask > 0].tolist() if int(v) > 0}
    assert len(circle_labs) == 1, circle_labs
    assert next(iter(circle_labs)) == ur


def test_ink_bridge_like_b_callout() -> None:
    """B-box ↔ vertical bar linked by a real ink arrow — stroke-cut severs it."""
    h, w = 160, 260
    binary = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(binary, (20, 50), (80, 110), 255, 2)
    cv2.putText(binary, "B", (38, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 2)
    cv2.line(binary, (200, 30), (200, 130), 255, 3)
    cv2.line(binary, (80, 80), (200, 80), 255, 2)
    cv2.arrowedLine(binary, (170, 80), (200, 80), 255, 2, tipLength=0.35)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(binary, k)
    _, cc = cv2.connectedComponents(dilated)
    main = int(cc[80, 20])
    assert main > 0 and int(cc[80, 200]) == main

    label_map = (cc == main).astype(np.int32)
    blocks = [_block(1, label_map == 1)]
    new_map, _, meta = split_by_stroke(
        label_map, blocks, binary, [(140, 55), (140, 105)], stroke_width=5
    )
    assert meta["ok"], meta
    _assert_stroke_mode(meta)
    left = int(new_map[80, 20])
    right = int(new_map[80, 200])
    assert left != right and left > 0 and right > 0


def test_enclosed_inner_box() -> None:
    """Inner box glued to outer frame by dilation — cut the local neck."""
    h, w = 200, 200
    binary = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(binary, (20, 20), (180, 180), 255, 3)
    cv2.rectangle(binary, (35, 80), (70, 120), 255, -1)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(binary, k)
    _, cc = cv2.connectedComponents(dilated)
    main = int(cc[100, 50])
    assert main > 0 and int(cc[20, 100]) == main

    label_map = (cc == main).astype(np.int32)
    blocks = [_block(1, label_map == 1)]
    new_map, _, meta = split_by_stroke(
        label_map, blocks, binary, [(28, 70), (28, 130)], stroke_width=5
    )
    assert meta["ok"], meta
    _assert_stroke_mode(meta)
    assert len(meta["new_labels"]) == 2
    inner = int(new_map[100, 50])
    outer_r = int(new_map[100, 178])
    assert inner != outer_r and inner > 0 and outer_r > 0


def test_solid_ink_body() -> None:
    """Two blobs sharing solid ink (no dilation gap) — hard stroke cut."""
    h, w = 100, 180
    binary = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(binary, (10, 30), (80, 70), 255, -1)
    cv2.rectangle(binary, (80, 40), (160, 60), 255, -1)
    _, cc = cv2.connectedComponents(binary)
    main = int(cc[50, 40])
    assert main > 0 and int(cc[50, 120]) == main
    label_map = (cc == main).astype(np.int32)
    blocks = [_block(1, label_map == 1)]
    new_map, _, meta = split_by_stroke(
        label_map, blocks, binary, [(80, 25), (80, 75)], stroke_width=5
    )
    assert meta["ok"], meta
    _assert_stroke_mode(meta)
    assert int(new_map[50, 40]) != int(new_map[50, 120])


def test_arrow_bridge_hard_cut() -> None:
    """Stroke across a real ink arrow bridge must hard-cut into two pieces."""
    h, w = 160, 280
    binary = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(binary, (20, 50), (80, 110), 255, 2)
    cv2.putText(binary, "B", (38, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 2)
    cv2.line(binary, (200, 30), (200, 130), 255, 3)
    cv2.line(binary, (80, 80), (200, 80), 255, 2)
    cv2.arrowedLine(binary, (170, 80), (200, 80), 255, 2, tipLength=0.35)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(binary, k)
    _, cc = cv2.connectedComponents(dilated)
    main = int(cc[80, 20])
    assert main > 0 and int(cc[80, 200]) == main
    label_map = (cc == main).astype(np.int32)
    blocks = [_block(1, label_map == 1)]
    new_map, _, meta = split_by_stroke(
        label_map, blocks, binary, [(140, 55), (140, 105)], stroke_width=5
    )
    assert meta["ok"], meta
    _assert_stroke_mode(meta)
    assert int(new_map[80, 20]) != int(new_map[80, 200])
    keep, small = meta["new_labels"]
    assert keep == main or keep == 1
    assert small != keep


def test_largest_keeps_color() -> None:
    """After a successful cut, the larger piece keeps the original label."""
    h, w = 100, 220
    binary = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(binary, (10, 20), (120, 80), 255, -1)  # large
    cv2.rectangle(binary, (150, 30), (200, 70), 255, -1)  # small
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dilated = cv2.dilate(binary, k)
    label_map = (dilated > 0).astype(np.int32)
    blocks = [_block(1, label_map == 1)]
    new_map, _, meta = split_by_stroke(
        label_map, blocks, binary, [(135, 15), (135, 85)], stroke_width=5
    )
    assert meta["ok"], meta
    keep, small = meta["new_labels"]
    assert keep == 1 and small == 2
    assert int(new_map[50, 60]) == 1
    assert int(new_map[50, 175]) == 2


def test_twin_views_scale_label_new_color() -> None:
    """Stroke on ``M1:2`` peels the label to a new color; twin views keep original."""
    h, w = 200, 360
    binary = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(binary, (20, 40), (140, 170), 255, -1)
    cv2.rectangle(binary, (220, 40), (340, 170), 255, -1)
    cv2.putText(binary, "M1:2", (155, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    dilated = cv2.dilate(binary, k)
    _, cc = cv2.connectedComponents(dilated)
    left = int(cc[100, 80])
    right = int(cc[100, 280])
    assert left > 0 and left == right

    text_mask = np.zeros_like(binary)
    cv2.putText(text_mask, "M1:2", (155, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    tys, txs = np.where(text_mask > 0)
    ty, tx = int(tys[len(tys) // 2]), int(txs[len(txs) // 2])
    assert int(cc[ty, tx]) == left

    label_map = (cc == left).astype(np.int32)
    blocks = [_block(1, label_map == 1)]
    # Stroke across the scale text (user intent: peel M1:2 with new color).
    new_map, _, meta = split_by_stroke(
        label_map,
        blocks,
        binary,
        [(tx - 8, ty - 2), (tx + 24, ty + 2)],
        stroke_width=5,
    )
    assert meta["ok"], meta
    keep, small = meta["new_labels"]
    assert keep == 1 and small != 1
    assert str(meta["snap"].get("mode", "")).startswith("stroke_peel"), meta["snap"]
    # Twin views stay on the original color together.
    assert int(new_map[100, 80]) == keep
    assert int(new_map[100, 280]) == keep
    # Scale text gets the new color.
    assert int(new_map[ty, tx]) == small


if __name__ == "__main__":
    test_two_rects_glued()
    test_upper_gap_distant_circle_untouched()
    test_ink_bridge_like_b_callout()
    test_enclosed_inner_box()
    test_solid_ink_body()
    test_arrow_bridge_hard_cut()
    test_largest_keeps_color()
    test_twin_views_scale_label_new_color()
    print("smoke_local_halo_split: ALL PASS")
