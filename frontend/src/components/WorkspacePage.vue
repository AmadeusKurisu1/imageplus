<template>
  <div class="content-grid" :class="{ 'split-mode': viewMode === 'split' && previewUrl && resultUrl }">
    <ImageCanvas
      :viewMode="viewMode"
      :previewUrl="previewUrl"
      :resultUrl="resultUrl"
      :resultMeta="resultMeta"
      :selectedFile="selectedFile"
      :processing="processing"
      :statusText="statusText"
      :statusKind="statusKind"
      :elapsedSeconds="elapsedSeconds"
      :splitPosition="splitPosition"
      :jobs="jobs"
      :activePreviewUrl="activePreviewUrl"
      @update:viewMode="$emit('update:viewMode', $event)"
      @openViewer="$emit('openViewer', $event)"
      @fileSelected="$emit('fileSelected', $event)"
      @batchFilesSelected="$emit('batchFilesSelected', $event)"
      @update:splitPosition="$emit('update:splitPosition', $event)"
      @restoreJob="$emit('restoreJob', $event)"
    />

    <ParameterInspector
      :models="models"
      :selectedModel="selectedModel"
      :settings="settings"
      :processing="processing"
      :selectedFile="selectedFile"
      @refreshHealth="$emit('refreshHealth')"
      @syncModelDefaults="$emit('syncModelDefaults')"
      @runEnhance="$emit('runEnhance')"
      @clearAll="$emit('clearAll')"
    />

    <BatchQueue
      v-if="batchQueue.length > 0"
      :batchQueue="batchQueue"
      :batchProcessing="batchProcessing"
      @startBatch="$emit('startBatch')"
      @cancelBatch="$emit('cancelBatch')"
      @clearQueue="$emit('clearQueue')"
      @removeFromQueue="$emit('removeFromQueue', $event)"
      @viewResult="$emit('viewResult', $event)"
    />
  </div>
</template>

<script setup>
import ImageCanvas from "./ImageCanvas.vue";
import ParameterInspector from "./ParameterInspector.vue";
import BatchQueue from "./BatchQueue.vue";

defineProps({
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
  models: { type: Array, default: () => [] },
  selectedModel: { type: Object, default: null },
  settings: { type: Object, required: true },
  batchQueue: { type: Array, default: () => [] },
  batchProcessing: { type: Boolean, default: false },
});

defineEmits([
  "update:viewMode", "openViewer", "fileSelected", "batchFilesSelected",
  "update:splitPosition", "restoreJob", "refreshHealth", "syncModelDefaults",
  "runEnhance", "clearAll", "startBatch", "cancelBatch", "clearQueue",
  "removeFromQueue", "viewResult",
]);
</script>
