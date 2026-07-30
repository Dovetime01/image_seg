<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import type { ToolMode } from '../api'

const props = defineProps<{
  imageUrl: string
  overlayUrl: string | null
  width: number
  height: number
  mode: ToolMode
  disabled?: boolean
}>()

const emit = defineEmits<{
  stroke: [points: number[][]]
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const wrapRef = ref<HTMLDivElement | null>(null)

const scale = ref(1)
const offsetX = ref(0)
const offsetY = ref(0)
const drawing = ref(false)
const panning = ref(false)
const lastPan = ref({ x: 0, y: 0 })
const strokePts = ref<{ x: number; y: number }[]>([])

const baseImg = ref<HTMLImageElement | null>(null)
const overlayImg = ref<HTMLImageElement | null>(null)

const cursor = computed(() => {
  if (props.mode === 'pan') return panning.value ? 'grabbing' : 'grab'
  if (props.mode === 'delete') return 'pointer'
  return 'crosshair'
})

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Failed to load ${url}`))
    img.src = url
  })
}

async function reloadBase() {
  if (!props.imageUrl) {
    baseImg.value = null
    return
  }
  baseImg.value = await loadImage(props.imageUrl)
  fitToView()
  draw()
}

async function reloadOverlay() {
  if (!props.overlayUrl) {
    overlayImg.value = null
    draw()
    return
  }
  try {
    overlayImg.value = await loadImage(props.overlayUrl)
  } catch {
    overlayImg.value = null
  }
  draw()
}

function fitToView() {
  const wrap = wrapRef.value
  if (!wrap || !props.width || !props.height) return
  const pad = 24
  const sw = (wrap.clientWidth - pad) / props.width
  const sh = (wrap.clientHeight - pad) / props.height
  scale.value = Math.min(sw, sh, 1.5)
  offsetX.value = (wrap.clientWidth - props.width * scale.value) / 2
  offsetY.value = (wrap.clientHeight - props.height * scale.value) / 2
}

function screenToImage(sx: number, sy: number) {
  return {
    x: (sx - offsetX.value) / scale.value,
    y: (sy - offsetY.value) / scale.value,
  }
}

function strokeColor() {
  if (props.mode === 'split') return '#ff4d4f'
  if (props.mode === 'merge') return '#52c41a'
  if (props.mode === 'delete') return '#faad14'
  return '#3d8bfd'
}

function draw() {
  const canvas = canvasRef.value
  const wrap = wrapRef.value
  if (!canvas || !wrap) return
  const dpr = window.devicePixelRatio || 1
  const w = wrap.clientWidth
  const h = wrap.clientHeight
  canvas.width = Math.floor(w * dpr)
  canvas.height = Math.floor(h * dpr)
  canvas.style.width = `${w}px`
  canvas.style.height = `${h}px`
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)
  ctx.fillStyle = '#1a1d23'
  ctx.fillRect(0, 0, w, h)

  ctx.save()
  ctx.translate(offsetX.value, offsetY.value)
  ctx.scale(scale.value, scale.value)

  if (overlayImg.value) {
    ctx.drawImage(overlayImg.value, 0, 0, props.width, props.height)
  } else if (baseImg.value) {
    ctx.drawImage(baseImg.value, 0, 0, props.width, props.height)
  } else {
    ctx.fillStyle = '#2a2f3a'
    ctx.fillRect(0, 0, props.width, props.height)
  }

  if (strokePts.value.length >= 1) {
    ctx.strokeStyle = strokeColor()
    ctx.fillStyle = strokeColor()
    ctx.lineWidth = Math.max(2, 3 / scale.value)
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    if (strokePts.value.length === 1) {
      const p = strokePts.value[0]
      ctx.beginPath()
      ctx.arc(p.x, p.y, Math.max(4, 6 / scale.value), 0, Math.PI * 2)
      ctx.fill()
    } else {
      ctx.beginPath()
      ctx.moveTo(strokePts.value[0].x, strokePts.value[0].y)
      for (let i = 1; i < strokePts.value.length; i++) {
        ctx.lineTo(strokePts.value[i].x, strokePts.value[i].y)
      }
      ctx.stroke()
    }
  }
  ctx.restore()
}

function onWheel(e: WheelEvent) {
  e.preventDefault()
  // Trackpad two-finger scroll / mouse wheel without ctrl → pan.
  // Pinch-zoom on macOS Chrome sets ctrlKey (and often metaKey).
  const isPinchZoom = e.ctrlKey || e.metaKey
  if (!isPinchZoom) {
    offsetX.value -= e.deltaX
    offsetY.value -= e.deltaY
    draw()
    return
  }

  const rect = canvasRef.value!.getBoundingClientRect()
  const mx = e.clientX - rect.left
  const my = e.clientY - rect.top
  const before = screenToImage(mx, my)
  const factor = e.deltaY < 0 ? 1.1 : 0.9
  scale.value = Math.min(8, Math.max(0.05, scale.value * factor))
  offsetX.value = mx - before.x * scale.value
  offsetY.value = my - before.y * scale.value
  draw()
}

function onPointerDown(e: PointerEvent) {
  if (props.disabled) return
  const canvas = canvasRef.value!
  canvas.setPointerCapture(e.pointerId)
  const rect = canvas.getBoundingClientRect()
  const sx = e.clientX - rect.left
  const sy = e.clientY - rect.top

  if (props.mode === 'pan' || e.button === 1 || e.altKey) {
    panning.value = true
    lastPan.value = { x: sx, y: sy }
    return
  }

  drawing.value = true
  const pt = screenToImage(sx, sy)
  strokePts.value = [pt]
  draw()
}

function onPointerMove(e: PointerEvent) {
  const canvas = canvasRef.value!
  const rect = canvas.getBoundingClientRect()
  const sx = e.clientX - rect.left
  const sy = e.clientY - rect.top

  if (panning.value) {
    offsetX.value += sx - lastPan.value.x
    offsetY.value += sy - lastPan.value.y
    lastPan.value = { x: sx, y: sy }
    draw()
    return
  }

  if (!drawing.value) return
  const pt = screenToImage(sx, sy)
  const last = strokePts.value[strokePts.value.length - 1]
  const dx = pt.x - last.x
  const dy = pt.y - last.y
  if (dx * dx + dy * dy >= 4) {
    strokePts.value.push(pt)
    draw()
  }
}

function finishStroke() {
  if (!drawing.value) return
  drawing.value = false
  const pts = strokePts.value
  strokePts.value = []
  draw()
  // Delete mode allows a single click; other edit modes need a stroke.
  if (props.mode === 'delete' && pts.length >= 1) {
    emit(
      'stroke',
      pts.map((p) => [p.x, p.y]),
    )
    return
  }
  if (pts.length >= 2) {
    emit(
      'stroke',
      pts.map((p) => [p.x, p.y]),
    )
  }
}

function onPointerUp() {
  if (panning.value) {
    panning.value = false
    return
  }
  finishStroke()
}

function onResize() {
  fitToView()
  draw()
}

watch(() => props.imageUrl, reloadBase)
watch(() => props.overlayUrl, reloadOverlay)
watch(
  () => [props.width, props.height],
  () => {
    fitToView()
    draw()
  },
)

onMounted(() => {
  window.addEventListener('resize', onResize)
  reloadBase().then(reloadOverlay)
})

onUnmounted(() => {
  window.removeEventListener('resize', onResize)
})

defineExpose({ fitToView, draw })
</script>

<template>
  <div ref="wrapRef" class="viewer-wrap">
    <canvas
      ref="canvasRef"
      class="viewer-canvas"
      :style="{ cursor }"
      @wheel.prevent="onWheel"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointercancel="onPointerUp"
      @contextmenu.prevent
    />
  </div>
</template>

<style scoped>
.viewer-wrap {
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: hidden;
  background: #12151a;
  border-radius: 8px;
}
.viewer-canvas {
  width: 100%;
  height: 100%;
  display: block;
  touch-action: none;
}
</style>
