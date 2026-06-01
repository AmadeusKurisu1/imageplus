<template>
  <section class="history-strip" aria-label="任务记录">
    <div class="history-title">最近任务</div>
    <button v-for="job in jobs" :key="job.id" class="history-item" type="button" @click="$emit('restoreJob', job)">
      <img :src="job.thumb" alt="" />
      <span>{{ job.modelTitle }}</span>
      <small>{{ job.elapsed }}</small>
    </button>
    <div v-if="jobs.length === 0" class="history-empty">完成一次超分后会在这里保留快捷记录</div>
  </section>
</template>

<script setup>
defineProps({
  jobs: { type: Array, default: () => [] },
});

defineEmits(["restoreJob"]);
</script>
