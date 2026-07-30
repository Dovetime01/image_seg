export type ToolMode = 'pan' | 'split' | 'merge' | 'delete'


export interface PageSummary {
  index: number
  name: string
  source_file: string
  width: number
  height: number
  analyzed: boolean
  dirty: boolean
  block_count: number
  can_undo: boolean
  debug_steps?: string[]
  meta: Record<string, unknown>
}

export interface BlockInfo {
  id: number
  x: number
  y: number
  w: number
  h: number
  area: number
  cc_label: number
  group_label: number
}

export interface AnalyzeParams {
  gap_thres: number
  text_ocr_refine: boolean
  text_gap_thres: number
  alpha: number
}

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json()
    return data.detail || JSON.stringify(data)
  } catch {
    return res.statusText || `HTTP ${res.status}`
  }
}

export async function createJob(files: FileList | File[]): Promise<{
  job_id: string
  page_count: number
  pages: PageSummary[]
  errors: string[]
}> {
  const form = new FormData()
  for (const f of Array.from(files)) {
    form.append('files', f)
  }
  const res = await fetch('/api/jobs?pdf_dpi=150', { method: 'POST', body: form })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function getJob(jobId: string) {
  const res = await fetch(`/api/jobs/${jobId}`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<{ job_id: string; pages: PageSummary[] }>
}

export function pageImageUrl(jobId: string, pageIndex: number) {
  return `/api/jobs/${jobId}/pages/${pageIndex}/image`
}

export function pageThumbUrl(jobId: string, pageIndex: number) {
  return `/api/jobs/${jobId}/pages/${pageIndex}/thumbnail`
}

export function pageOverlayUrl(jobId: string, pageIndex: number, bust = 0) {
  return `/api/jobs/${jobId}/pages/${pageIndex}/overlay?t=${bust}`
}

export async function analyzePage(jobId: string, pageIndex: number, params: AnalyzeParams) {
  const res = await fetch(`/api/jobs/${jobId}/pages/${pageIndex}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<{
    meta: Record<string, unknown>
    summary: PageSummary
    blocks: BlockInfo[]
  }>
}

async function readNdjsonStream(
  res: Response,
  onProgress?: (step: string, message: string) => void,
): Promise<Record<string, unknown>> {
  if (!res.ok) throw new Error(await parseError(res))
  if (!res.body) throw new Error('Empty stream body')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let result: Record<string, unknown> | null = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed) continue
      const evt = JSON.parse(trimmed) as {
        type: string
        step?: string
        message?: string
        [key: string]: unknown
      }
      if (evt.type === 'progress') {
        onProgress?.(evt.step || '', evt.message || '')
      } else if (evt.type === 'error') {
        throw new Error(String(evt.message || 'Analyze failed'))
      } else if (evt.type === 'result') {
        result = evt
      }
    }
  }

  if (buffer.trim()) {
    const evt = JSON.parse(buffer.trim()) as {
      type: string
      step?: string
      message?: string
      [key: string]: unknown
    }
    if (evt.type === 'progress') onProgress?.(evt.step || '', evt.message || '')
    else if (evt.type === 'error') throw new Error(String(evt.message || 'Analyze failed'))
    else if (evt.type === 'result') result = evt
  }

  if (!result) throw new Error('Stream ended without result')
  return result
}

export async function analyzePageStream(
  jobId: string,
  pageIndex: number,
  params: AnalyzeParams,
  onProgress?: (step: string, message: string) => void,
) {
  const res = await fetch(`/api/jobs/${jobId}/pages/${pageIndex}/analyze-stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  const evt = await readNdjsonStream(res, onProgress)
  return {
    meta: (evt.meta || {}) as Record<string, unknown>,
    summary: evt.summary as PageSummary,
    blocks: (evt.blocks || []) as BlockInfo[],
  }
}

export async function analyzeAll(jobId: string, params: AnalyzeParams) {
  const res = await fetch(`/api/jobs/${jobId}/analyze-all`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<{ pages: PageSummary[]; results: unknown[] }>
}

export async function analyzeAllStream(
  jobId: string,
  params: AnalyzeParams,
  onProgress?: (step: string, message: string) => void,
) {
  const res = await fetch(`/api/jobs/${jobId}/analyze-all-stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  const evt = await readNdjsonStream(res, onProgress)
  return {
    pages: (evt.pages || []) as PageSummary[],
    results: (evt.results || []) as unknown[],
  }
}

export async function splitStroke(
  jobId: string,
  pageIndex: number,
  points: number[][],
  strokeWidth = 4,
) {
  const res = await fetch(`/api/jobs/${jobId}/pages/${pageIndex}/split`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ points, stroke_width: strokeWidth }),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function mergeStroke(
  jobId: string,
  pageIndex: number,
  points: number[][],
  strokeWidth = 8,
) {
  const res = await fetch(`/api/jobs/${jobId}/pages/${pageIndex}/merge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ points, stroke_width: strokeWidth }),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function deleteStroke(
  jobId: string,
  pageIndex: number,
  points: number[][],
  strokeWidth = 10,
) {
  const res = await fetch(`/api/jobs/${jobId}/pages/${pageIndex}/delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ points, stroke_width: strokeWidth }),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function undoPage(jobId: string, pageIndex: number) {
  const res = await fetch(`/api/jobs/${jobId}/pages/${pageIndex}/undo`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export function exportOverlayUrl(jobId: string, pageIndex: number) {
  return `/api/jobs/${jobId}/pages/${pageIndex}/export/overlay.png`
}

export function exportPartsUrl(jobId: string, pageIndex: number) {
  return `/api/jobs/${jobId}/pages/${pageIndex}/export/parts.zip`
}

export function pageDebugUrl(jobId: string, pageIndex: number, stepId: string, bust = 0) {
  return `/api/jobs/${jobId}/pages/${pageIndex}/debug/${stepId}?t=${bust}`
}

export async function listDebugSteps(jobId: string, pageIndex: number) {
  const res = await fetch(`/api/jobs/${jobId}/pages/${pageIndex}/debug-steps`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json() as Promise<{ steps: { id: string; label: string }[]; analyzed: boolean }>
}
