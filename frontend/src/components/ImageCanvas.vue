<template>
  <section class="canvas-area">
    <div class="toolbar">
      <div class="segmented" role="tablist" aria-label="查看模式">
        <button :class="{ active: viewMode === 'compare' }" type="button" @click="$emit('update:viewMode', 'compare')">对比</button>
        <button :class="{ active: viewMode === 'split' }" type="button" @click="$emit('update:viewMode', 'split')" :disabled="!previewUrl || !resultUrl">滑块</button>
        <button :class="{ active: viewMode === 'input' }" type="button" @click="$emit('update:viewMode', 'input')">原图</button>
        <button :class="{ active: viewMode === 'output' }" type="button" @click="$emit('update:viewMode', 'output')">结果</button>
      </div>
      <div class="toolbar-actions">
        <button class="icon-button" type="button" :disabled="!activePreviewUrl" title="放大查看" @click="$emit('openViewer', activePreviewUrl)">
          <svg viewBox="0 0 24 24" fill="none"><path d="M8 3H3v5M21 8V3h-5M16 21h5v-5M3 16v5h5" stroke="currentColor" stroke-width="2" stroke-linecap="round" /></svg>
        </button>
        <a class="icon-button" :class="{ disabled: !resultUrl }" :href="resultUrl || '#'" title="下载结果" download>
          <svg viewBox="0 0 24 24" fill="none"><path d="M12 3v12m0 0 4-4m-4 4-4-4M5 21h14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
        </a>
      </div>
    </div>

    <div class="viewer-grid" :class="viewMode">
      <article class="image-card input-card" @dragover.prevent @drop.prevent="handleDrop">
        <div class="image-card-header">
          <span>原图</span>
          <small>{{ selectedFile?.name || "等待上传" }}</small>
        </div>
        <button v-if="!previewUrl" class="drop-zone" type="button" @click="$refs.fileInput?.click()">
          <svg viewBox="0 0 32 32" fill="none"><path d="M16 21V7m0 0-5 5m5-5 5 5M8 24h16" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" /></svg>
          <span>拖入图片或点击上传</span>
          <small>PNG / JPG / WebP，支持透明通道</small>
        </button>
        <button v-else class="image-frame" type="button" @click="$emit('openViewer', previewUrl)">
          <img :src="previewUrl" alt="原图预览" />
        </button>
      </article>

      <article class="image-card output-card">
        <div class="image-card-header">
          <span>增强结果</span>
          <small>{{ resultMeta || "处理完成后显示" }}</small>
        </div>
        <button v-if="!resultUrl" class="empty-result" type="button" disabled>
          <svg viewBox="0 0 32 32" fill="none"><path d="M7 22l6-7 4 4 3-4 5 7M9 8h14a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V10a2 2 0 0 1 2-2Zm11 5h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
          <span>{{ processing ? "正在增强..." : "暂无结果" }}</span>
        </button>
        <button v-else class="image-frame" type="button" @click="$emit('openViewer', resultUrl)">
          <img :src="resultUrl" alt="增强结果预览" />
        </button>
      </article>
    </div>

    <div v-if="viewMode === 'split' && previewUrl && resultUrl" class="split-view" @mousedown.prevent="startSplitDrag" @touchstart.prevent="startSplitDrag">
      <div class="split-container" ref="splitContainerRef">
        <img :src="resultUrl" class="split-under" alt="增强结果" />
        <div class="split-over" :style="{ clipPath: `inset(0 ${100 - localSplitPosition}% 0 0)` }">
          <img :src="previewUrl" alt="原图" />
        </div>
        <div class="split-handle" :style="{ left: localSplitPosition + '%' }">
          <div class="split-handle-line"></div>
        </div>
        <span class="split-label split-label-left">原图</span>
        <span class="split-label split-label-right">增强结果</span>
      </div>
    </div>

    <div class="status-strip" :class="{ error: statusKind === 'error', success: statusKind === 'success' }">
      <span>{{ statusText }}</span>
      <span v-if="processing" class="pulse">处理中 {{ elapsedSeconds }}s</span>
    </div>

    <HistoryStrip :jobs="jobs" @restore-job="$emit('restoreJob', $event)" />

    <input ref="fileInput" class="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" multiple @change="handleFileInput" />
  </section>
</template>

<script setup>
import { ref, onUnmounted } from "vue";
import HistoryStrip from "./HistoryStrip.vue";

const props = defineProps({
  viewMode: { type: String, default: "compare" },
  previewUrl: { type: String, default: "" },
  resultUrl: { type: String, default: "" },
  resultMeta: { type: String, default: "" },
  selectedFile: { type: Object, default: null },
  processing: { type: Boolean, default: false },
  statusText: { type: String, default: "" },
  statusKind: { type: String, default: "idle" },
  elapsedSeconds: { type: Number, default: 0 },
  splitPosition: { type: Number, default: 50 },
  jobs: { type: Array, default: () => [] },
  activePreviewUrl: { type: String, default: "" },
});

const emit = defineEmits([
  "update:viewMode", "openViewer", "fileSelected", "batchFilesSelected",
  "update:splitPosition", "restoreJob",
]);

const fileInput = ref(null);
const splitContainerRef = ref(null);
const localSplitPosition = ref(props.splitPosition);
let isDragging = false;

function handleFileInput(event) {
  const files = event.target.files;
  if (!files || files.length === 0) return;
  if (files.length === 1) {
    emit("fileSelected", files[0]);
  } else {
    emit("batchFilesSelected", Array.from(files));
  }
  if (fileInput.value) fileInput.value.value = "";
}

function handleDrop(event) {
  const files = event.dataTransfer.files;
  if (!files || files.length === 0) return;
  if (files.length === 1) {
    emit("fileSelected", files[0]);
  } else {
    emit("batchFilesSelected", Array.from(files));
  }
}

function startSplitDrag() {
  isDragging = true;
  document.addEventListener("mousemove", onSplitDrag);
  document.addEventListener("mouseup", stopSplitDrag);
  document.addEventListener("touchmove", onSplitDrag, { passive: false });
  document.addEventListener("touchend", stopSplitDrag);
}

function onSplitDrag(event) {
  if (!isDragging || !splitContainerRef.value) return;
  event.preventDefault();
  const rect = splitContainerRef.value.getBoundingClientRect();
  const x = event.touches ? event.touches[0].clientX : event.clientX;
  localSplitPosition.value = Math.max(2, Math.min(98, ((x - rect.left) / rect.width) * 100));
  emit("update:splitPosition", localSplitPosition.value);
}

function stopSplitDrag() {
  isDragging = false;
  document.removeEventListener("mousemove", onSplitDrag);
  document.removeEventListener("mouseup", stopSplitDrag);
  document.removeEventListener("touchmove", onSplitDrag);
  document.removeEventListener("touchend", stopSplitDrag);
}

onUnmounted(() => {
  stopSplitDrag();
});
</script>
