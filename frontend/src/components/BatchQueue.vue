<template>
  <div class="batch-queue">
    <div class="batch-header">
      <h3>批量队列 <small>({{ doneCount }}/{{ batchQueue.length }})</small></h3>
      <div class="batch-actions">
        <button class="small-button" type="button" :disabled="batchProcessing || batchQueue.length === 0" @click="$emit('startBatch')">
          {{ batchProcessing ? "处理中..." : "开始批量处理" }}
        </button>
        <button v-if="batchProcessing" class="small-button" type="button" @click="$emit('cancelBatch')">取消</button>
        <button class="small-button" type="button" :disabled="batchProcessing" @click="$emit('clearQueue')">清空队列</button>
      </div>
    </div>

    <div v-if="batchProcessing" class="batch-progress-bar">
      <div class="batch-progress-fill" :style="{ width: progressPercent + '%' }"></div>
    </div>

    <div class="batch-list">
      <div v-for="(item, index) in batchQueue" :key="item.id" class="batch-item">
        <span class="batch-item-name">{{ item.fileName }}</span>
        <span class="batch-status-badge" :class="item.status">{{ statusLabel(item.status) }}</span>
        <span class="batch-item-elapsed">{{ item.elapsed > 0 ? item.elapsed + 's' : '-' }}</span>
        <button v-if="item.status === 'done'" class="small-button" type="button" @click="$emit('viewResult', item)">查看</button>
        <button v-if="!batchProcessing && item.status !== 'processing'" class="small-button" type="button" @click="$emit('removeFromQueue', index)">移除</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  batchQueue: { type: Array, default: () => [] },
  batchProcessing: { type: Boolean, default: false },
});

defineEmits(["startBatch", "cancelBatch", "clearQueue", "removeFromQueue", "viewResult"]);

const doneCount = computed(() => props.batchQueue.filter(i => i.status === "done").length);
const progressPercent = computed(() => {
  if (props.batchQueue.length === 0) return 0;
  const done = props.batchQueue.filter(i => i.status === "done" || i.status === "error" || i.status === "cancelled").length;
  return Math.round((done / props.batchQueue.length) * 100);
});

function statusLabel(status) {
  const map = { pending: "等待中", processing: "处理中", done: "已完成", error: "失败", cancelled: "已取消" };
  return map[status] || status;
}
</script>
