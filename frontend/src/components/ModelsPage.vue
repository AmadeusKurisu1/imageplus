<template>
  <section class="feature-page models-page">
    <div class="page-header">
      <div>
        <h2>模型管理</h2>
        <p>查看权重状态、模型特性，并快速切换到工作台。</p>
      </div>
      <button class="small-button" type="button" @click="$emit('refreshHealth')">刷新状态</button>
    </div>

    <div class="runtime-cards" aria-label="运行环境">
      <article>
        <span>推理设备</span>
        <strong>{{ health?.device?.label || "检测中" }}</strong>
      </article>
      <article>
        <span>PyTorch</span>
        <strong>{{ health?.torch || "-" }}</strong>
      </article>
      <article>
        <span>CUDA</span>
        <strong>{{ health?.cuda || "N/A" }}</strong>
      </article>
    </div>

    <div v-if="models.length === 0" class="empty-state">
      <strong>正在读取模型列表</strong>
      <span>如果长时间没有变化，检查后端是否正常启动。</span>
    </div>

    <div v-else class="model-grid">
      <article
        v-for="model in models"
        :key="model.key"
        class="model-card"
        :class="{ active: model.key === settings.model_key }"
      >
        <div class="model-card-head">
          <div>
            <h3>{{ model.title }}</h3>
            <p>{{ model.description }}</p>
          </div>
          <span class="status-pill" :class="{ ready: model.ready }">
            {{ model.ready ? "已就绪" : "首次运行下载" }}
          </span>
        </div>
        <div class="model-specs">
          <span>{{ model.arch }}</span>
          <span>标称 {{ model.scale }}x</span>
          <span>{{ model.supports_denoise ? "支持降噪" : "固定权重" }}</span>
        </div>
        <div class="model-actions">
          <button class="secondary-button" type="button" @click="$emit('chooseModel', model)">
            {{ model.key === settings.model_key ? "回到工作台" : "使用此模型" }}
          </button>
        </div>
      </article>
    </div>
  </section>
</template>

<script setup>
defineProps({
  models: { type: Array, default: () => [] },
  settings: { type: Object, required: true },
  health: { type: Object, default: null },
});

defineEmits(["refreshHealth", "chooseModel"]);
</script>
