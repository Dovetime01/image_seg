"""FastAPI server for the info-block segmentation demo."""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

# Project root (image_seg/) on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.edit_ops import delete_by_stroke, merge_by_stroke, split_by_stroke
from backend.pipeline import (
    DEBUG_STEP_DEFS,
    analyze_page,
    blocks_to_json,
    encode_jpeg,
    export_parts_zip,
    make_thumbnail,
    page_summary,
    pages_from_upload,
    refresh_overlay,
)
from backend.store import PageState, store

app = FastAPI(title="Info-block Segmentation Demo", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeBody(BaseModel):
    gap_thres: int = 6
    text_ocr_refine: bool = False
    text_gap_thres: int = 10
    alpha: float = 0.34


class StrokeBody(BaseModel):
    points: List[List[float]] = Field(
        ..., description="Polyline in image pixel coords [[x,y], ...]"
    )
    stroke_width: int = 4


class AlphaBody(BaseModel):
    alpha: float = 0.40


def _job(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


def _page(job_id: str, page_index: int) -> PageState:
    job = _job(job_id)
    if page_index < 0 or page_index >= len(job.pages):
        raise HTTPException(404, f"Page not found: {page_index}")
    return job.pages[page_index]


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/jobs")
async def create_job(
    files: List[UploadFile] = File(...),
    pdf_dpi: int = 150,
):
    if not files:
        raise HTTPException(400, "No files uploaded")

    job = store.create()
    page_index = 0
    errors: List[str] = []

    for uf in files:
        raw = await uf.read()
        name = uf.filename or "upload.bin"
        try:
            pages = pages_from_upload(name, raw, pdf_dpi=pdf_dpi)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            continue
        job.created_files.append(name)
        for page_name, img in pages:
            page = PageState(
                index=page_index,
                name=page_name,
                source_file=name,
                image_bgr=img,
                thumbnail_jpeg=make_thumbnail(img),
            )
            job.pages.append(page)
            page_index += 1

    if not job.pages:
        store.delete(job.id)
        raise HTTPException(400, f"No pages loaded. Errors: {errors}")

    return {
        "job_id": job.id,
        "page_count": len(job.pages),
        "files": job.created_files,
        "pages": [page_summary(p) for p in job.pages],
        "errors": errors,
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _job(job_id)
    return {
        "job_id": job.id,
        "page_count": len(job.pages),
        "files": job.created_files,
        "pages": [page_summary(p) for p in job.pages],
    }


@app.get("/api/jobs/{job_id}/pages/{page_index}/image")
def get_page_image(job_id: str, page_index: int, quality: int = 90):
    page = _page(job_id, page_index)
    return Response(
        content=encode_jpeg(page.image_bgr, quality=quality),
        media_type="image/jpeg",
    )


@app.get("/api/jobs/{job_id}/pages/{page_index}/thumbnail")
def get_page_thumbnail(job_id: str, page_index: int):
    page = _page(job_id, page_index)
    data = page.thumbnail_jpeg or make_thumbnail(page.image_bgr)
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/pages/{page_index}/overlay")
def get_overlay(job_id: str, page_index: int):
    page = _page(job_id, page_index)
    if page.overlay_png is None:
        raise HTTPException(400, "Page not analyzed yet")
    return Response(content=page.overlay_png, media_type="image/png")


@app.get("/api/jobs/{job_id}/pages/{page_index}/debug-steps")
def list_debug_steps(job_id: str, page_index: int):
    page = _page(job_id, page_index)
    steps = []
    for step_id, label in DEBUG_STEP_DEFS:
        if step_id in page.debug_images:
            steps.append({"id": step_id, "label": label})
    return {"steps": steps, "analyzed": page.analyzed}


@app.get("/api/jobs/{job_id}/pages/{page_index}/debug/{step_id}")
def get_debug_image(job_id: str, page_index: int, step_id: str):
    page = _page(job_id, page_index)
    data = page.debug_images.get(step_id)
    if data is None:
        raise HTTPException(404, f"Debug step not found: {step_id}")
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/pages/{page_index}/blocks")
def get_blocks(job_id: str, page_index: int):
    page = _page(job_id, page_index)
    return {
        "blocks": blocks_to_json(page),
        "summary": page_summary(page),
    }


@app.post("/api/jobs/{job_id}/pages/{page_index}/analyze")
def analyze_one(job_id: str, page_index: int, body: AnalyzeBody):
    page = _page(job_id, page_index)
    try:
        meta = analyze_page(
            page,
            gap_thres=body.gap_thres,
            text_ocr_refine=body.text_ocr_refine,
            text_gap_thres=body.text_gap_thres,
            alpha=body.alpha,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Analyze failed: {exc}") from exc
    return {
        "meta": meta,
        "summary": page_summary(page),
        "blocks": blocks_to_json(page),
    }


@app.post("/api/jobs/{job_id}/pages/{page_index}/analyze-stream")
def analyze_one_stream(job_id: str, page_index: int, body: AnalyzeBody):
    """NDJSON stream: progress lines then a final result line."""
    page = _page(job_id, page_index)

    def event_gen():
        q: queue.Queue = queue.Queue()

        def on_progress(step: str, message: str) -> None:
            q.put({"type": "progress", "step": step, "message": message})

        def worker() -> None:
            try:
                on_progress("start", f"开始分析第 {page_index + 1} 页…")
                meta = analyze_page(
                    page,
                    gap_thres=body.gap_thres,
                    text_ocr_refine=body.text_ocr_refine,
                    text_gap_thres=body.text_gap_thres,
                    alpha=body.alpha,
                    progress=on_progress,
                )
                q.put(
                    {
                        "type": "result",
                        "meta": meta,
                        "summary": page_summary(page),
                        "blocks": blocks_to_json(page),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                q.put({"type": "error", "message": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(event_gen(), media_type="application/x-ndjson")


@app.post("/api/jobs/{job_id}/analyze-all")
def analyze_all(job_id: str, body: AnalyzeBody):
    job = _job(job_id)
    results = []
    for page in job.pages:
        try:
            meta = analyze_page(
                page,
                gap_thres=body.gap_thres,
                text_ocr_refine=body.text_ocr_refine,
                text_gap_thres=body.text_gap_thres,
                alpha=body.alpha,
            )
            results.append({"index": page.index, "ok": True, "meta": meta})
        except Exception as exc:  # noqa: BLE001
            results.append({"index": page.index, "ok": False, "error": str(exc)})
    return {
        "pages": [page_summary(p) for p in job.pages],
        "results": results,
    }


@app.post("/api/jobs/{job_id}/analyze-all-stream")
def analyze_all_stream(job_id: str, body: AnalyzeBody):
    job = _job(job_id)

    def event_gen():
        q: queue.Queue = queue.Queue()

        def on_progress(step: str, message: str) -> None:
            q.put({"type": "progress", "step": step, "message": message})

        def worker() -> None:
            results = []
            try:
                for page in job.pages:
                    on_progress(
                        "page_start",
                        f"正在分析第 {page.index + 1}/{len(job.pages)} 页（{page.name}）…",
                    )
                    try:
                        meta = analyze_page(
                            page,
                            gap_thres=body.gap_thres,
                            text_ocr_refine=body.text_ocr_refine,
                            text_gap_thres=body.text_gap_thres,
                            alpha=body.alpha,
                            progress=on_progress,
                        )
                        results.append(
                            {"index": page.index, "ok": True, "meta": meta}
                        )
                    except Exception as exc:  # noqa: BLE001
                        results.append(
                            {"index": page.index, "ok": False, "error": str(exc)}
                        )
                        on_progress(
                            "page_error",
                            f"第 {page.index + 1} 页失败：{exc}",
                        )
                q.put(
                    {
                        "type": "result",
                        "pages": [page_summary(p) for p in job.pages],
                        "results": results,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                q.put({"type": "error", "message": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(event_gen(), media_type="application/x-ndjson")


@app.post("/api/jobs/{job_id}/pages/{page_index}/split")
def split_page(job_id: str, page_index: int, body: StrokeBody):
    page = _page(job_id, page_index)
    if page.label_map is None:
        raise HTTPException(400, "Analyze the page first")
    if page.binary is None:
        raise HTTPException(400, "Missing binary mask; re-analyze the page")

    page.push_undo()
    points = [(float(p[0]), float(p[1])) for p in body.points if len(p) >= 2]
    new_map, new_blocks, meta = split_by_stroke(
        page.label_map,
        page.blocks,
        page.binary,
        points,
        stroke_width=body.stroke_width,
    )
    if not meta.get("ok"):
        page.undo_stack.pop()  # discard empty undo
        return {"ok": False, "meta": meta, "summary": page_summary(page)}

    page.label_map = new_map
    page.blocks = new_blocks
    page.dirty = True
    refresh_overlay(page)
    page.last_meta = {**page.last_meta, "last_edit": meta}
    return {
        "ok": True,
        "meta": meta,
        "summary": page_summary(page),
        "blocks": blocks_to_json(page),
    }


@app.post("/api/jobs/{job_id}/pages/{page_index}/merge")
def merge_page(job_id: str, page_index: int, body: StrokeBody):
    page = _page(job_id, page_index)
    if page.label_map is None:
        raise HTTPException(400, "Analyze the page first")
    if page.binary is None:
        raise HTTPException(400, "Missing binary mask; re-analyze the page")

    page.push_undo()
    points = [(float(p[0]), float(p[1])) for p in body.points if len(p) >= 2]
    new_map, new_blocks, meta = merge_by_stroke(
        page.label_map,
        page.blocks,
        page.binary,
        points,
        stroke_width=max(body.stroke_width, 6),
    )
    if not meta.get("ok"):
        page.undo_stack.pop()
        return {"ok": False, "meta": meta, "summary": page_summary(page)}

    page.label_map = new_map
    page.blocks = new_blocks
    page.dirty = True
    refresh_overlay(page)
    page.last_meta = {**page.last_meta, "last_edit": meta}
    return {
        "ok": True,
        "meta": meta,
        "summary": page_summary(page),
        "blocks": blocks_to_json(page),
    }


@app.post("/api/jobs/{job_id}/pages/{page_index}/delete")
def delete_page_blocks(job_id: str, page_index: int, body: StrokeBody):
    page = _page(job_id, page_index)
    if page.label_map is None:
        raise HTTPException(400, "Analyze the page first")

    page.push_undo()
    points = [(float(p[0]), float(p[1])) for p in body.points if len(p) >= 2]
    new_map, new_blocks, meta = delete_by_stroke(
        page.label_map,
        page.blocks,
        points,
        hit_radius=max(body.stroke_width, 8),
    )
    if not meta.get("ok"):
        page.undo_stack.pop()
        return {"ok": False, "meta": meta, "summary": page_summary(page)}

    page.label_map = new_map
    page.blocks = new_blocks
    page.dirty = True
    refresh_overlay(page)
    # Keep overlay debug preview roughly in sync for the menu.
    if page.overlay_png and "overlay" in page.debug_images:
        try:
            import cv2
            import numpy as np

            arr = np.frombuffer(page.overlay_png, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                from backend.pipeline import encode_jpeg

                page.debug_images["overlay"] = encode_jpeg(img, quality=80)
        except Exception:
            pass
    page.last_meta = {**page.last_meta, "last_edit": meta}
    return {
        "ok": True,
        "meta": meta,
        "summary": page_summary(page),
        "blocks": blocks_to_json(page),
    }


@app.post("/api/jobs/{job_id}/pages/{page_index}/undo")
def undo_page(job_id: str, page_index: int):
    page = _page(job_id, page_index)
    if not page.undo():
        raise HTTPException(400, "Nothing to undo")
    return {"ok": True, "summary": page_summary(page), "blocks": blocks_to_json(page)}


@app.post("/api/jobs/{job_id}/pages/{page_index}/alpha")
def set_alpha(job_id: str, page_index: int, body: AlphaBody):
    page = _page(job_id, page_index)
    if page.label_map is None:
        raise HTTPException(400, "Analyze the page first")
    refresh_overlay(page, alpha=body.alpha)
    return {"ok": True, "summary": page_summary(page)}


def _ascii_filename(name: str, suffix: str) -> str:
    """HTTP headers must be latin-1; keep downloads ASCII-safe."""
    safe = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "-_") else "_" for ch in name)
    safe = safe.strip("_") or "page"
    return f"{safe}{suffix}"


@app.get("/api/jobs/{job_id}/pages/{page_index}/export/overlay.png")
def export_overlay(job_id: str, page_index: int):
    page = _page(job_id, page_index)
    if page.overlay_png is None:
        raise HTTPException(400, "Analyze the page first")
    filename = _ascii_filename(page.name, "_overlay.png")
    return Response(
        content=page.overlay_png,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/jobs/{job_id}/pages/{page_index}/export/parts.zip")
def export_parts(job_id: str, page_index: int):
    page = _page(job_id, page_index)
    try:
        data = export_parts_zip(page)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    filename = _ascii_filename(page.name, "_parts.zip")
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def main():
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=_ROOT,
    )


if __name__ == "__main__":
    main()
