<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { listDebugSteps, pageDebugUrl } from '../api'

const props = defineProps<{
  open: boolean
  jobId: string
  pageIndex: number
  pageName: string
}>()

const emit = defineEmits<{
  close: []
}>()

const steps = ref<{ id: string; label: string }[]>([])
const active = ref('')
const bust = ref(0)
const loading = ref(false)
const error = ref('')

const imageUrl = computed(() => {
  if (!props.jobId || !active.value) return ''
  return pageDebugUrl(props.jobId, props.pageIndex, active.value, bust.value)
})

async function loadSteps() {
  if (!props.open || !props.jobId) return
  loading.value = true
  error.value = ''
  try {
    const res = await listDebugSteps(props.jobId, props.pageIndex)
    steps.value = res.steps
    active.value = res.steps[0]?.id || ''
    bust.value += 1
    if (!res.steps.length) error.value = '暂无过程图，请先分析该页'
  } catch (err) {
    error.value = (err as Error).message
    steps.value = []
  } finally {
    loading.value = false
  }
}

watch(
  () => [props.open, props.jobId, props.pageIndex],
  () => {
    if (props.open) void loadSteps()
  },
)
</script>

<template>
  <div v-if="open" class="mask" @click.self="emit('close')">
    <div class="panel">
      <header class="head">
        <div>
          <h2>过程图</h2>
          <p>{{ pageName }} · 第 {{ pageIndex + 1 }} 页</p>
        </div>
        <button type="button" class="close" @click="emit('close')">关闭</button>
      </header>

      <div class="body">
        <aside class="steps">
          <button
            v-for="s in steps"
            :key="s.id"
            type="button"
            class="step"
            :class="{ active: active === s.id }"
            @click="active = s.id; bust += 1"
          >
            {{ s.label }}
          </button>
          <p v-if="loading" class="hint">加载中…</p>
          <p v-if="error" class="err">{{ error }}</p>
        </aside>
        <div class="preview">
          <img v-if="imageUrl" :src="imageUrl" :alt="active" />
          <div v-else class="empty">选择左侧步骤查看</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.mask {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  display: grid;
  place-items: center;
  z-index: 50;
  padding: 24px;
}
.panel {
  width: min(1100px, 96vw);
  height: min(780px, 90vh);
  background: #151921;
  border: 1px solid #334155;
  border-radius: 12px;
  display: grid;
  grid-template-rows: auto 1fr;
  overflow: hidden;
}
.head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 16px;
  border-bottom: 1px solid #273041;
}
.head h2 {
  margin: 0;
  font-size: 16px;
}
.head p {
  margin: 4px 0 0;
  color: #8b95a8;
  font-size: 12px;
}
.close {
  border: 1px solid #334155;
  background: #1c2430;
  color: #e7ecf5;
  border-radius: 8px;
  padding: 6px 12px;
  cursor: pointer;
}
.body {
  display: grid;
  grid-template-columns: 200px 1fr;
  min-height: 0;
}
.steps {
  border-right: 1px solid #273041;
  padding: 10px;
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.step {
  border: 1px solid #2f3748;
  background: #1c2430;
  color: #d7dde8;
  border-radius: 8px;
  padding: 8px 10px;
  text-align: left;
  cursor: pointer;
  font-size: 12px;
}
.step.active {
  border-color: #3d8bfd;
  background: #243b63;
}
.preview {
  min-width: 0;
  min-height: 0;
  background: #0e1116;
  display: grid;
  place-items: center;
  padding: 12px;
}
.preview img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: 6px;
}
.empty,
.hint {
  color: #8b95a8;
  font-size: 13px;
}
.err {
  color: #f0a0a0;
  font-size: 12px;
}
</style>
