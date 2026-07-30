"""Info-block segmentation via OpenCV dilate + connected components.

Nearby foreground ink is merged into text/info blocks by controlling the
dilation kernel size (gap threshold).
"""

from .extract import InfoBlock, ExtractConfig, extract_info_blocks
from .export_parts import export_part_crops
from .visualize import (
    draw_blocks,
    draw_region_overlay,
    draw_region_overlay_from_debug,
    save_debug_bundle,
)

__all__ = [
    "InfoBlock",
    "ExtractConfig",
    "extract_info_blocks",
    "export_part_crops",
    "draw_blocks",
    "draw_region_overlay",
    "draw_region_overlay_from_debug",
    "save_debug_bundle",
]
