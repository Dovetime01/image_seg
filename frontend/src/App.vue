<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import {
  analyzeAllStream,
  analyzePageStream,
  createJob,
  deleteStroke,
  exportOverlayUrl,
  exportPartsUrl,
  mergeStroke,
  pageImageUrl,
  pageOverlayUrl,
  splitStroke,
  undoPage,
  type AnalyzeParams,
  type PageSummary,
  type ToolMode,
} from './api'
import PageStrip from './components/PageStrip.vue'
import ViewerCanvas from './components/ViewerCanvas.vue'
import DebugStepsModal from './components/DebugStepsModal.vue'

const jobId = ref<string | null>(null)
const pages = ref<PageSummary[]>([])
const current = ref(0)
const mode = ref<ToolMode>('pan')
const busy = ref(false)
const status = ref('导入 PDF / PNG / JPG 开始')
const pipelineSteps = ref<{ step: string; message: string; done: boolean }[]>([])
const overlayBust = ref(0)

const gapThres = ref(6)
const textRefine = ref(false)
const textGap = ref(10)
const alpha = ref(0.34)

/** Per-page analysis params. UI edits apply to the current page only. */
const pageParams = ref<Record<number, AnalyzeParams>>({})
/** Suppress "please re-analyze" hint while loading another page's params into the UI. */
let loadingPageParams = false

const fileInput = ref<HTMLInputElement | null>(null)
const debugOpen = ref(false)
const debugPageIndex = ref(0)

function defaultParams(): AnalyzeParams {
  return {
    gap_thres: 6,
    text_ocr_refine: false,
    text_gap_thres: 10,
    alpha: 0.34,
  }
}

function readUiParams(): AnalyzeParams {
  return {
    gap_thres: gapThres.value,
    text_ocr_refine: textRefine.value,
    text_gap_thres: textGap.value,
    alpha: alpha.value,
  }
}

function writeUiParams(p: AnalyzeParams) {
  loadingPageParams = true
  gapThres.value = p.gap_thres
  textRefine.value = p.text_ocr_refine
  textGap.value = p.text_gap_thres
  alpha.value = p.alpha
  // Next tick would be nicer; microtask is enough to skip the watch hint once.
  queueMicrotask(() => {
    loadingPageParams = false
  })
}

function ensurePageParams(index: number): AnalyzeParams {
  if (!pageParams.value[index]) {
    pageParams.value[index] = defaultParams()
  }
  return pageParams.value[index]
}

function saveCurrentPageParams() {
  if (!jobId.value) return
  pageParams.value[current.value] = readUiParams()
}

function loadPageParamsToUi(index: number) {
  writeUiParams(ensurePageParams(index))
}

function syncParamsToAllPages(p: AnalyzeParams) {
  const next: Record<number, AnalyzeParams> = { ...pageParams.value }
  for (const page of pages.value) {
    next[page.index] = { ...p }
  }
  pageParams.value = next
}

const currentPage = computed(() => pages.value.find((p) => p.index === current.value) || null)

const modeHint = computed(() => {
  switch (mode.value) {
    case 'pan':
      return '浏览：拖拽平移画布，滚轮缩放；双指滑动平移，捏合缩放。'
    case 'split':
      return '拆分：在粘连缝隙处画一刀，系统沿膨胀桥接中线切开（尽量不伤笔划）。双指滑动平移，捏合缩放。'
    case 'merge':
      return '融合：画一条线穿过要合并的两个（或多个）区域，系统将它们合并为一块。双指滑动平移，捏合缩放。'
    case 'delete':
      return '删除：点击或画线选中误检区域，系统将其删除。双指滑动平移，捏合缩放。'
    default:
      return ''
  }
})

const imageUrl = computed(() =>
  jobId.value && currentPage.value ? pageImageUrl(jobId.value, current.value) : '',
)

const overlayUrl = computed(() => {
  if (!jobId.value || !currentPage.value?.analyzed) return null
  return pageOverlayUrl(jobId.value, current.value, overlayBust.value)
})

function bumpOverlay() {
  overlayBust.value += 1
}

function updatePageSummary(summary: PageSummary) {
  const idx = pages.value.findIndex((p) => p.index === summary.index)
  if (idx >= 0) pages.value[idx] = summary
  else pages.value.push(summary)
}

function resetPipeline() {
  pipelineSteps.value = []
}

function pushProgress(step: string, message: string) {
  status.value = message
  const last = pipelineSteps.value[pipelineSteps.value.length - 1]
  if (last && !last.done) last.done = true
  pipelineSteps.value.push({ step, message, done: false })
  // Keep recent steps only
  if (pipelineSteps.value.length > 14) {
    pipelineSteps.value = pipelineSteps.value.slice(-14)
  }
}

function finishPipeline(finalMessage: string) {
  const last = pipelineSteps.value[pipelineSteps.value.length - 1]
  if (last) last.done = true
  status.value = finalMessage
}

async function onFiles(ev: Event) {
  const input = ev.target as HTMLInputElement
  if (!input.files?.length) return
  busy.value = true
  resetPipeline()
  const names = Array.from(input.files).map((f) => f.name)
  const hasPdf = names.some((n) => n.toLowerCase().endsWith('.pdf'))
  pushProgress('import', hasPdf ? '正在导入并渲染 PDF…' : '正在导入图片…')
  try {
    const res = await createJob(input.files)
    jobId.value = res.job_id
    pages.value = res.pages
    current.value = 0
    // Init independent params for every page from current UI defaults.
    const base = readUiParams()
    const init: Record<number, AnalyzeParams> = {}
    for (const page of res.pages) {
      init[page.index] = { ...base }
    }
    pageParams.value = init
    writeUiParams(base)
    finishPipeline(
      `已导入 ${res.page_count} 页` +
        (res.errors?.length ? `（部分失败：${res.errors.join('; ')}）` : ''),
    )
    await runAnalyzeCurrent()
  } catch (err) {
    status.value = `导入失败：${(err as Error).message}`
  } finally {
    busy.value = false
    input.value = ''
  }
}

async function runAnalyzeCurrent() {
  if (!jobId.value || !currentPage.value) return
  saveCurrentPageParams()
  const pageCfg = ensurePageParams(current.value)
  writeUiParams(pageCfg)
  busy.value = true
  resetPipeline()
  pushProgress('start', `开始分析第 ${current.value + 1} 页…`)
  try {
    const res = await analyzePageStream(jobId.value, current.value, pageCfg, pushProgress)
    updatePageSummary(res.summary)
    bumpOverlay()
    finishPipeline(`第 ${current.value + 1} 页：${res.summary.block_count} 个区域`)
  } catch (err) {
    status.value = `分析失败：${(err as Error).message}`
  } finally {
    busy.value = false
  }
}

async function runAnalyzeAll() {
  if (!jobId.value) return
  // Current UI params become the global set for every page.
  const globalCfg = readUiParams()
  syncParamsToAllPages(globalCfg)
  writeUiParams(globalCfg)
  busy.value = true
  resetPipeline()
  pushProgress('batch', '已同步当前参数到全部页，开始批量分析…')
  try {
    const res = await analyzeAllStream(jobId.value, globalCfg, pushProgress)
    pages.value = res.pages
    bumpOverlay()
    finishPipeline(
      `全部完成：${res.pages.filter((p) => p.analyzed).length}/${res.pages.length} 页（已用当前参数）`,
    )
  } catch (err) {
    status.value = `批量分析失败：${(err as Error).message}`
  } finally {
    busy.value = false
  }
}

function scheduleHint() {
  if (!jobId.value || loadingPageParams) return
  status.value = '当前页参数已修改，请点击「重新分析当前页」；或「分析全部页」同步到所有页'
}

watch([gapThres, textRefine, textGap, alpha], () => {
  if (loadingPageParams) return
  if (!jobId.value) return
  saveCurrentPageParams()
  if (currentPage.value?.analyzed) scheduleHint()
})

async function selectPage(index: number) {
  // Persist edits on the page we're leaving.
  saveCurrentPageParams()
  current.value = index
  loadPageParamsToUi(index)
  const page = pages.value.find((p) => p.index === index)
  if (page && !page.analyzed) {
    await runAnalyzeCurrent()
  } else {
    bumpOverlay()
    status.value = `已切换到第 ${index + 1} 页参数`
  }
}

function openDebug(index: number) {
  debugPageIndex.value = index
  debugOpen.value = true
}

const debugPageName = computed(
  () => pages.value.find((p) => p.index === debugPageIndex.value)?.name || '',
)

async function onStroke(points: number[][]) {
  if (!jobId.value || busy.value) return
  if (mode.value !== 'split' && mode.value !== 'merge' && mode.value !== 'delete') return
  busy.value = true
  status.value =
    mode.value === 'split' ? '拆分中…' : mode.value === 'merge' ? '融合中…' : '删除中…'
  try {
    const fn =
      mode.value === 'split' ? splitStroke : mode.value === 'merge' ? mergeStroke : deleteStroke
    const res = await fn(jobId.value, current.value, points)
    if (res.summary) updatePageSummary(res.summary)
    bumpOverlay()
    if (res.ok) {
      status.value = res.meta?.reason || '编辑成功'
    } else {
      status.value = res.meta?.reason || '未生效'
    }
  } catch (err) {
    status.value = `编辑失败：${(err as Error).message}`
  } finally {
    busy.value = false
  }
}

async function onUndo() {
  if (!jobId.value) return
  busy.value = true
  try {
    const res = await undoPage(jobId.value, current.value)
    if (res.summary) updatePageSummary(res.summary)
    bumpOverlay()
    status.value = '已撤销'
  } catch (err) {
    status.value = `撤销失败：${(err as Error).message}`
  } finally {
    busy.value = false
  }
}

async function downloadBlob(url: string, filename: string) {
  const res = await fetch(url)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const data = await res.json()
      detail = data.detail || JSON.stringify(data)
    } catch {
      /* ignore */
    }
    throw new Error(detail || `HTTP ${res.status}`)
  }
  const blob = await res.blob()
  const obj = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = obj
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(obj)
}

async function exportOverlay() {
  if (!jobId.value || !currentPage.value) return
  busy.value = true
  status.value = '正在导出效果图…'
  try {
    const name = `${currentPage.value.name || 'page'}_overlay.png`.replace(/[^\w.\-]+/g, '_')
    await downloadBlob(exportOverlayUrl(jobId.value, current.value), name)
    status.value = '效果图已导出'
  } catch (err) {
    status.value = `导出失败：${(err as Error).message}`
  } finally {
    busy.value = false
  }
}

async function exportParts() {
  if (!jobId.value || !currentPage.value) return
  busy.value = true
  status.value = '正在导出组件 ZIP…'
  try {
    const name = `${currentPage.value.name || 'page'}_parts.zip`.replace(/[^\w.\-]+/g, '_')
    await downloadBlob(exportPartsUrl(jobId.value, current.value), name)
    status.value = '组件 ZIP 已导出'
  } catch (err) {
    status.value = `导出失败：${(err as Error).message}`
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <h1>信息块分割</h1>
        <p>Canvas 审核</p>
      </div>

      <section class="sec">
        <button class="btn primary" type="button" :disabled="busy" @click="fileInput?.click()">
          导入 PDF / 图片
        </button>
        <input
          ref="fileInput"
          type="file"
          multiple
          accept=".pdf,.png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp"
          hidden
          @change="onFiles"
        />
      </section>

      <section class="sec">
        <div class="label">当前页参数（第 {{ current + 1 }} 页）</div>
        <label class="row">
          <span class="row-title">
            文字二次膨胀
            <span
              class="info-tip"
              tabindex="0"
              role="img"
              aria-label="文字二次膨胀说明"
              @click.prevent
            >
              <span class="info-tip__btn" aria-hidden="true">i</span>
              <span class="info-tip__bubble" role="tooltip">
                用于把被拆散的文字/注记（如尺寸、比例、标题）用更大膨胀核再合并成完整文字块，避免字被切成碎块。视图、图框等非文字区域不受影响。
                <br />
                <strong>备注：勾选此项后需重新分析当页。</strong>
              </span>
            </span>
          </span>
          <input v-model="textRefine" type="checkbox" :disabled="busy || !jobId" />
        </label>

        <label class="field">
          <span>膨胀核 / 区域边距 {{ gapThres }}（需点重新分析）</span>
          <input
            v-model.number="gapThres"
            type="range"
            min="1"
            max="40"
            :disabled="busy || !jobId"
          />
        </label>

        <label v-if="textRefine" class="field">
          <span>文字膨胀核 {{ textGap }}（需点重新分析）</span>
          <input
            v-model.number="textGap"
            type="range"
            min="2"
            max="40"
            :disabled="busy || !jobId"
          />
        </label>

        <label class="field">
          <span>叠色透明度 {{ alpha.toFixed(2) }}（需点重新分析）</span>
          <input
            v-model.number="alpha"
            type="range"
            min="0.1"
            max="0.8"
            step="0.02"
            :disabled="busy || !jobId"
          />
        </label>
        <p class="hint">改滑条只影响当前页；「分析全部页」会把当前参数同步到所有页再批量分析。</p>
      </section>

      <section class="sec">
        <div class="label">审核模式</div>
        <div class="modes">
          <button
            type="button"
            class="btn"
            :class="{ active: mode === 'pan' }"
            @click="mode = 'pan'"
          >
            浏览
          </button>
          <button
            type="button"
            class="btn"
            :class="{ active: mode === 'split' }"
            :disabled="!currentPage?.analyzed"
            @click="mode = 'split'"
          >
            拆分
          </button>
          <button
            type="button"
            class="btn"
            :class="{ active: mode === 'merge' }"
            :disabled="!currentPage?.analyzed"
            @click="mode = 'merge'"
          >
            融合
          </button>
          <button
            type="button"
            class="btn"
            :class="{ active: mode === 'delete' }"
            :disabled="!currentPage?.analyzed"
            @click="mode = 'delete'"
          >
            删除
          </button>
        </div>
        <p class="hint">{{ modeHint }}</p>
      </section>

      <section class="sec actions">
        <button class="btn" type="button" :disabled="busy || !jobId" @click="runAnalyzeCurrent">
          重新分析当前页
        </button>
        <button class="btn" type="button" :disabled="busy || !jobId" @click="runAnalyzeAll">
          分析全部页（同步参数）
        </button>
        <button
          class="btn"
          type="button"
          :disabled="busy || !currentPage?.can_undo"
          @click="onUndo"
        >
          撤销
        </button>
        <button
          class="btn"
          type="button"
          :disabled="!currentPage?.analyzed"
          @click="exportOverlay"
        >
          导出效果图 PNG
        </button>
        <button class="btn" type="button" :disabled="!currentPage?.analyzed" @click="exportParts">
          导出各组件 ZIP
        </button>
      </section>

      <div class="status">
        <div class="status-main">{{ status }}</div>
        <ul v-if="pipelineSteps.length" class="pipeline">
          <li
            v-for="(s, i) in pipelineSteps"
            :key="`${s.step}-${i}`"
            :class="{ done: s.done, current: !s.done && i === pipelineSteps.length - 1 }"
          >
            <span class="dot" />
            <span>{{ s.message }}</span>
          </li>
        </ul>
      </div>
    </aside>

    <main class="main">
      <div class="viewer">
        <ViewerCanvas
          v-if="jobId && currentPage"
          :image-url="imageUrl"
          :overlay-url="overlayUrl"
          :width="currentPage.width"
          :height="currentPage.height"
          :mode="mode"
          :disabled="busy"
          @stroke="onStroke"
        />
        <div v-else class="empty">拖入或点击左侧导入工程图（支持批量 / 多页 PDF）</div>
      </div>
      <PageStrip
        v-if="jobId && pages.length"
        :job-id="jobId"
        :pages="pages"
        :current="current"
        @select="selectPage"
        @open-debug="openDebug"
      />
    </main>

    <DebugStepsModal
      v-if="jobId"
      :open="debugOpen"
      :job-id="jobId"
      :page-index="debugPageIndex"
      :page-name="debugPageName"
      @close="debugOpen = false"
    />
  </div>
</template>

<style scoped>
.app {
  display: grid;
  grid-template-columns: 300px 1fr;
  height: 100vh;
  background: #0f1217;
  color: #e7ecf5;
  font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
}

.sidebar {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 16px;
  border-right: 1px solid #273041;
  background: #151921;
  overflow: auto;
}

.brand h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 650;
  letter-spacing: 0.02em;
}
.brand p {
  margin: 4px 0 0;
  color: #8b95a8;
  font-size: 12px;
}

.sec {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-top: 8px;
  border-top: 1px solid #243043;
}

.row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
}

.row-title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.info-tip {
  position: relative;
  display: inline-flex;
  align-items: center;
  flex-shrink: 0;
  cursor: help;
}

.info-tip__btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  border: 1px solid #6b7a90;
  color: #9aa6b8;
  font-size: 10px;
  font-weight: 700;
  font-style: italic;
  line-height: 1;
  user-select: none;
}

.info-tip__bubble {
  display: none;
  position: absolute;
  left: 0;
  top: calc(100% + 8px);
  z-index: 40;
  width: 168px;
  max-width: calc(100vw - 48px);
  padding: 8px 10px;
  border-radius: 8px;
  border: 1px solid #334155;
  background: #1c2430;
  color: #d5dbe6;
  font-size: 11px;
  font-style: normal;
  font-weight: 400;
  line-height: 1.5;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
  pointer-events: none;
}

.info-tip__bubble strong {
  color: #f0c674;
  font-weight: 600;
}

.info-tip:hover .info-tip__btn,
.info-tip:focus-within .info-tip__btn {
  border-color: #60a5fa;
  color: #93c5fd;
}

.info-tip:hover .info-tip__bubble,
.info-tip:focus-within .info-tip__bubble {
  display: block;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 12px;
  color: #b7c0d0;
}
.field input[type='range'] {
  width: 100%;
}

.label {
  font-size: 12px;
  color: #9aa6b8;
}

.modes {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
}

.hint {
  margin: 0;
  font-size: 11px;
  line-height: 1.45;
  color: #7f8a9d;
}

.actions .btn {
  width: 100%;
}

.btn {
  border: 1px solid #334155;
  background: #1c2430;
  color: #e7ecf5;
  border-radius: 8px;
  padding: 8px 10px;
  font-size: 13px;
  cursor: pointer;
}
.btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.btn.primary {
  background: #2563eb;
  border-color: #2563eb;
}
.btn.active {
  background: #243b63;
  border-color: #3d8bfd;
}

.status {
  margin-top: auto;
  font-size: 12px;
  color: #9eb0c9;
  line-height: 1.4;
  padding: 10px;
  background: #10151d;
  border-radius: 8px;
  border: 1px solid #243043;
}
.status-main {
  font-weight: 600;
  color: #d7e0ef;
  margin-bottom: 6px;
}
.pipeline {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 220px;
  overflow: auto;
}
.pipeline li {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  color: #7f8a9d;
}
.pipeline li.done {
  color: #7dcea0;
}
.pipeline li.current {
  color: #9ec5ff;
}
.pipeline .dot {
  width: 6px;
  height: 6px;
  margin-top: 5px;
  border-radius: 50%;
  background: currentColor;
  flex: 0 0 auto;
}

.main {
  display: grid;
  grid-template-rows: 1fr auto;
  min-width: 0;
  min-height: 0;
}

.viewer {
  min-height: 0;
  padding: 12px;
}

.empty {
  height: 100%;
  display: grid;
  place-items: center;
  color: #7f8a9d;
  border: 1px dashed #334155;
  border-radius: 8px;
  background: #12161d;
}
</style>
