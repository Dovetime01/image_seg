"""Info-block segmentation for the Web review demo."""

from .extract import InfoBlock, ExtractConfig, extract_info_blocks
from .export_parts import export_part_crops
from .visualize import (
    draw_region_overlay,
    draw_region_overlay_from_debug,
)

__all__ = [
    "InfoBlock",
    "ExtractConfig",
    "extract_info_blocks",
    "export_part_crops",
    "draw_region_overlay",
    "draw_region_overlay_from_debug",
]
