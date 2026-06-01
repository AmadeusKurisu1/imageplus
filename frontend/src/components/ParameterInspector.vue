<template>
  <aside class="inspector" aria-label="处理参数">
    <div class="panel-heading">
      <div>
        <h2>处理参数</h2>
        <p>模型会按需加载并缓存</p>
      </div>
      <button class="small-button" type="button" @click="$emit('refreshHealth')">刷新</button>
    </div>

    <label class="field">
      <span>模型</span>
      <select v-model="settings.model_key" @change="$emit('syncModelDefaults')">
        <option v-for="model in models" :key="model.key" :value="model.key">{{ model.title }}</option>
      </select>
    </label>

    <div class="model-note">
      <strong>{{ selectedModel?.title || "模型" }}</strong>
      <p>{{ selectedModel?.description || "加载模型信息中。" }}</p>
      <small>
        {{ selectedModel?.arch || "-" }} · 输出 {{ settings.outscale }}x
        <template v-if="settings.outscale !== selectedModel?.scale">（模型标称 {{ selectedModel?.scale }}x，自动适配）</template>
        · {{ selectedModel?.supports_denoise ? "支持降噪" : "固定权重" }}
      </small>
    </div>

    <label class="field">
      <span>输出倍率 <b>{{ settings.outscale }}x</b></span>
      <input v-model.number="settings.outscale" type="range" min="1" max="4" step="0.5" />
    </label>

    <label class="field">
      <span>分块大小</span>
      <select v-model.number="settings.tile_size">
        <option :value="0">关闭</option>
        <option :value="128">128 px</option>
        <option :value="256">256 px</option>
        <option :value="512">512 px</option>
      </select>
    </label>

    <label class="field" :class="{ disabled: !selectedModel?.supports_denoise }">
      <span>降噪强度 <b>{{ settings.denoise_strength.toFixed(2) }}</b></span>
      <input v-model.number="settings.denoise_strength" type="range" min="0" max="1" step="0.05" :disabled="!selectedModel?.supports_denoise" />
    </label>

    <label class="field">
      <span>透明通道处理</span>
      <select v-model="settings.alpha_upsampler">
        <option value="realesrgan">Real-ESRGAN</option>
        <option value="bicubic">Bicubic</option>
      </select>
    </label>

    <div class="toggle-list">
      <label>
        <input v-model="settings.face_enhance" type="checkbox" />
        <span>人脸增强</span>
      </label>
      <label>
        <input v-model="settings.use_fp32" type="checkbox" />
        <span>FP32 精度</span>
      </label>
      <label>
        <input v-model="settings.compute_metrics" type="checkbox" />
        <span>质量评估 (PSNR/SSIM)</span>
      </label>
    </div>

    <button class="run-button" type="button" :disabled="!selectedFile || processing" @click="$emit('runEnhance')">
      <span>{{ processing ? "处理中..." : "开始超分" }}</span>
      <svg viewBox="0 0 24 24" fill="none"><path d="M5 12h14m0 0-5-5m5 5-5 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
    </button>
    <button class="clear-button" type="button" @click="$emit('clearAll')">清空工作区</button>
  </aside>
</template>

<script setup>
defineProps({
  models: { type: Array, default: () => [] },
  selectedModel: { type: Object, default: null },
  settings: { type: Object, required: true },
  processing: { type: Boolean, default: false },
  selectedFile: { type: Object, default: null },
});

defineEmits(["refreshHealth", "syncModelDefaults", "runEnhance", "clearAll"]);
</script>
