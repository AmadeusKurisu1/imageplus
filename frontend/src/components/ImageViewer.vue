<template>
  <div class="lightbox" role="dialog" aria-modal="true" @click.self="$emit('close')">
    <div class="lightbox-toolbar">
      <span>{{ viewerTitle }}</span>
      <div>
        <button type="button" @click="$emit('zoomAt', zoom - 0.5)">-</button>
        <button type="button" @click="$emit('zoomAt', 1)">{{ Math.round(zoom * 100) }}%</button>
        <button type="button" @click="$emit('zoomAt', zoom + 0.5)">+</button>
        <button type="button" @click="$emit('reset')">重置</button>
        <button type="button" @click="$emit('close')">关闭</button>
      </div>
    </div>
    <div
      class="lightbox-canvas"
      :class="{ grabbing: viewerPanning }"
      @wheel.prevent="$emit('wheel', $event)"
      @mousedown="$emit('startPan', $event)"
      @mousemove="$emit('onPan', $event)"
      @mouseup="$emit('stopPan')"
      @mouseleave="$emit('stopPan')"
    >
      <img
        :src="viewerUrl"
        :style="{ transform: `translate(${offsetX}px, ${offsetY}px) scale(${zoom})` }"
        :draggable="false"
        alt="放大查看"
      />
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted } from "vue";

defineProps({
  viewerUrl: { type: String, required: true },
  viewerTitle: { type: String, default: "图片查看" },
  zoom: { type: Number, default: 1 },
  offsetX: { type: Number, default: 0 },
  offsetY: { type: Number, default: 0 },
  viewerPanning: { type: Boolean, default: false },
});

const emit = defineEmits(["close", "zoomAt", "reset", "wheel", "startPan", "onPan", "stopPan"]);

function onKeydown(event) {
  if (event.key === "Escape") emit("close");
}

onMounted(() => document.addEventListener("keydown", onKeydown));
onUnmounted(() => document.removeEventListener("keydown", onKeydown));
</script>
