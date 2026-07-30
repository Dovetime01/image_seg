<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import type { PageSummary } from '../api'
import { pageThumbUrl } from '../api'

const props = defineProps<{
  jobId: string
  pages: PageSummary[]
  current: number
}>()

const emit = defineEmits<{
  select: [index: number]
  openDebug: [index: number]
}>()

const menuOpenFor = ref<number | null>(null)

function toggleMenu(index: number, ev: MouseEvent) {
  ev.stopPropagation()
  menuOpenFor.value = menuOpenFor.value === index ? null : index
}

function onOpenDebug(index: number, ev: MouseEvent) {
  ev.stopPropagation()
  menuOpenFor.value = null
  emit('openDebug', index)
}

function onDocClick() {
  menuOpenFor.value = null
}

onMounted(() => document.addEventListener('click', onDocClick))
onUnmounted(() => document.removeEventListener('click', onDocClick))
</script>

<template>
  <div class="strip">
    <div v-for="p in pages" :key="p.index" class="thumb-wrap">
      <button
        class="thumb"
        :class="{ active: p.index === current }"
        type="button"
        @click="emit('select', p.index)"
      >
        <img :src="pageThumbUrl(jobId, p.index)" :alt="p.name" />
        <div class="meta">
          <span class="idx">{{ p.index + 1 }}</span>
          <span v-if="p.analyzed" class="badge ok">{{ p.block_count }}</span>
          <span v-else class="badge">…</span>
          <span v-if="p.dirty" class="badge edit">改</span>
        </div>
        <div class="name" :title="p.name">{{ p.name }}</div>
      </button>

      <div class="more-wrap">
        <button
          class="more-btn"
          type="button"
          title="更多"
          :disabled="!p.analyzed"
          @click="toggleMenu(p.index, $event)"
        >
          ⋯
        </button>
        <div v-if="menuOpenFor === p.index" class="menu" @click.stop>
          <button type="button" class="menu-item" @click="onOpenDebug(p.index, $event)">
            查看过程图
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.strip {
  display: flex;
  gap: 10px;
  overflow-x: auto;
  padding: 10px 12px;
  background: #161a21;
  border-top: 1px solid #2a3140;
}
.thumb-wrap {
  position: relative;
  flex: 0 0 auto;
}
.thumb {
  width: 120px;
  border: 1px solid #2f3748;
  background: #1c222c;
  border-radius: 8px;
  padding: 6px;
  color: #d7dde8;
  cursor: pointer;
  text-align: left;
}
.thumb.active {
  border-color: #3d8bfd;
  box-shadow: 0 0 0 1px #3d8bfd55;
}
.thumb img {
  width: 100%;
  height: 72px;
  object-fit: contain;
  background: #0e1116;
  border-radius: 4px;
  display: block;
}
.meta {
  display: flex;
  gap: 4px;
  align-items: center;
  margin-top: 6px;
  font-size: 11px;
}
.idx {
  font-weight: 600;
}
.badge {
  padding: 1px 5px;
  border-radius: 999px;
  background: #2a3140;
  color: #9aa6b8;
}
.badge.ok {
  background: #1f3d2f;
  color: #7dcea0;
}
.badge.edit {
  background: #3d2f1f;
  color: #e0b36a;
}
.name {
  margin-top: 2px;
  font-size: 11px;
  color: #8b95a8;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.more-wrap {
  position: absolute;
  top: 4px;
  right: 4px;
}
.more-btn {
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 6px;
  background: rgba(16, 21, 29, 0.82);
  color: #d7dde8;
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
  padding: 0;
}
.more-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}
.more-btn:hover:not(:disabled) {
  background: #243b63;
}
.menu {
  position: absolute;
  top: 28px;
  right: 0;
  min-width: 120px;
  background: #1c2430;
  border: 1px solid #334155;
  border-radius: 8px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
  z-index: 20;
  overflow: hidden;
}
.menu-item {
  display: block;
  width: 100%;
  border: none;
  background: transparent;
  color: #e7ecf5;
  text-align: left;
  padding: 8px 12px;
  font-size: 12px;
  cursor: pointer;
}
.menu-item:hover {
  background: #243b63;
}
</style>
