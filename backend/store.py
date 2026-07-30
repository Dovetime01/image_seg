"""In-memory job / page store for the segmentation demo."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from segm.extract import ExtractConfig, InfoBlock


@dataclass
class PageState:
    index: int
    name: str
    source_file: str
    image_bgr: np.ndarray
    thumbnail_jpeg: Optional[bytes] = None
    blocks: List[InfoBlock] = field(default_factory=list)
    label_map: Optional[np.ndarray] = None
    binary: Optional[np.ndarray] = None
    overlay_png: Optional[bytes] = None
    # Intermediate pipeline previews: step_id -> jpeg/png bytes
    debug_images: Dict[str, bytes] = field(default_factory=dict)
    analyzed: bool = False
    dirty: bool = False  # True after manual edit
    config: Optional[ExtractConfig] = None
    alpha: float = 0.40
    undo_stack: List[Tuple[np.ndarray, List[InfoBlock], bytes]] = field(
        default_factory=list
    )
    last_meta: Dict[str, Any] = field(default_factory=dict)

    def push_undo(self, max_steps: int = 20) -> None:
        if self.label_map is None or self.overlay_png is None:
            return
        snap = (
            self.label_map.copy(),
            [
                InfoBlock(
                    id=b.id,
                    x=b.x,
                    y=b.y,
                    w=b.w,
                    h=b.h,
                    area=b.area,
                    cc_label=b.cc_label,
                    group_label=b.group_label,
                )
                for b in self.blocks
            ],
            self.overlay_png,
        )
        self.undo_stack.append(snap)
        if len(self.undo_stack) > max_steps:
            self.undo_stack.pop(0)

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        label_map, blocks, overlay = self.undo_stack.pop()
        self.label_map = label_map
        self.blocks = blocks
        self.overlay_png = overlay
        self.dirty = True
        return True


@dataclass
class JobState:
    id: str
    pages: List[PageState] = field(default_factory=list)
    created_files: List[str] = field(default_factory=list)


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobState] = {}
        self._lock = threading.Lock()

    def create(self) -> JobState:
        job = JobState(id=uuid.uuid4().hex[:12])
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)


store = JobStore()
