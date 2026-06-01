<template>
  <section class="feature-page jobs-page">
    <div class="page-header">
      <div>
        <h2>任务记录</h2>
        <p>历史处理记录自动保存，刷新页面不丢失。</p>
      </div>
      <button class="small-button" type="button" :disabled="jobs.length === 0" @click="$emit('clearJobs')">清空记录</button>
    </div>

    <div v-if="jobs.length === 0" class="empty-state">
      <strong>还没有任务</strong>
      <span>回到工作台上传图片并完成一次超分后，这里会出现结果记录，刷新页面不会丢失。</span>
      <button class="secondary-button" type="button" @click="$emit('navigate', 'workspace')">去工作台</button>
    </div>

    <div v-else class="jobs-list">
      <article v-for="job in jobs" :key="job.id" class="job-row">
        <button class="job-thumb" type="button" @click="$emit('restoreJob', job)">
          <img :src="job.thumb" alt="" />
        </button>
        <div class="job-main">
          <h3>{{ job.modelTitle }}</h3>
          <p>{{ job.fileName }} · {{ job.createdAt }} · {{ job.elapsed }}</p>
          <span>{{ job.dimensions }}</span>
          <span v-if="job.psnr" class="metrics-badge">{{ job.psnr }} · {{ job.ssim }}</span>
        </div>
        <div class="job-actions">
          <button class="secondary-button" type="button" @click="$emit('restoreJob', job)">查看结果</button>
          <a class="text-button" :href="job.url" download>下载</a>
        </div>
      </article>
    </div>
  </section>
</template>

<script setup>
defineProps({
  jobs: { type: Array, default: () => [] },
});

defineEmits(["restoreJob", "clearJobs", "navigate"]);
</script>
