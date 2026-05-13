<template>
  <main class="app-shell">
    <aside class="sidebar" aria-label="工作区导航">
      <div class="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 32 32" fill="none">
          <path d="M7 9.5C7 8.1 8.1 7 9.5 7h13C23.9 7 25 8.1 25 9.5v13c0 1.4-1.1 2.5-2.5 2.5h-13A2.5 2.5 0 0 1 7 22.5v-13Z" stroke="currentColor" stroke-width="2" />
          <path d="M10.5 21.5 15 16l3.2 3.2 2.1-2.7 3.2 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
          <path d="M20.7 11.8h.1" stroke="currentColor" stroke-width="3" stroke-linecap="round" />
        </svg>
      </div>
      <nav class="nav-stack">
        <button
          class="nav-item"
          :class="{ active: activePage === 'workspace' }"
          type="button"
          title="增强工作台"
          aria-label="增强工作台"
          :aria-current="activePage === 'workspace' ? 'page' : undefined"
          @click="activePage = 'workspace'"
        >
          <svg viewBox="0 0 24 24" fill="none"><path d="M4 6h16M4 12h16M4 18h10" stroke="currentColor" stroke-width="2" stroke-linecap="round" /></svg>
          <span class="visually-hidden">增强工作台</span>
        </button>
        <button
          class="nav-item"
          :class="{ active: activePage === 'models' }"
          type="button"
          title="模型管理"
          aria-label="模型管理"
          :aria-current="activePage === 'models' ? 'page' : undefined"
          @click="activePage = 'models'"
        >
          <svg viewBox="0 0 24 24" fill="none"><path d="M12 3 4 7l8 4 8-4-8-4ZM4 12l8 4 8-4M4 17l8 4 8-4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
          <span class="visually-hidden">模型管理</span>
        </button>
        <button
          class="nav-item"
          :class="{ active: activePage === 'jobs' }"
          type="button"
          title="任务记录"
          aria-label="任务记录"
          :aria-current="activePage === 'jobs' ? 'page' : undefined"
          @click="activePage = 'jobs'"
        >
          <svg viewBox="0 0 24 24" fill="none"><path d="M8 7h10M8 12h10M8 17h6M4 7h.01M4 12h.01M4 17h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round" /></svg>
          <span class="visually-hidden">任务记录</span>
        </button>
      </nav>
      <div class="sidebar-status" :class="health?.device?.type || 'cpu'" title="推理设备">
        {{ deviceShortLabel }}
      </div>
    </aside>

    <section class="workspace">
      <header class="topbar">
        <div>
          <h1>{{ pageMeta.title }}</h1>
          <p>{{ pageMeta.description }}</p>
        </div>
        <div class="runtime">
          <span>{{ health?.device?.label || "检测中" }}</span>
          <span>PyTorch {{ health?.torch || "-" }}</span>
          <span>CUDA {{ health?.cuda || "N/A" }}</span>
        </div>
      </header>

      <div class="mobile-pages" role="tablist" aria-label="页面切换">
        <button
          v-for="page in pages"
          :key="page.id"
          :class="{ active: activePage === page.id }"
          type="button"
          @click="activePage = page.id"
        >
          {{ page.short }}
        </button>
      </div>

      <div v-if="activePage === 'workspace'" class="content-grid">
        <section class="canvas-area">
          <div class="toolbar">
            <div class="segmented" role="tablist" aria-label="查看模式">
              <button :class="{ active: viewMode === 'compare' }" type="button" @click="viewMode = 'compare'">对比</button>
              <button :class="{ active: viewMode === 'input' }" type="button" @click="viewMode = 'input'">原图</button>
              <button :class="{ active: viewMode === 'output' }" type="button" @click="viewMode = 'output'">结果</button>
            </div>
            <div class="toolbar-actions">
              <button class="icon-button" type="button" :disabled="!activePreviewUrl" title="放大查看" @click="openViewer(activePreviewUrl)">
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
              <button v-if="!previewUrl" class="drop-zone" type="button" @click="fileInput?.click()">
                <svg viewBox="0 0 32 32" fill="none"><path d="M16 21V7m0 0-5 5m5-5 5 5M8 24h16" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" /></svg>
                <span>拖入图片或点击上传</span>
                <small>PNG / JPG / WebP，支持透明通道</small>
              </button>
              <button v-else class="image-frame" type="button" @click="openViewer(previewUrl)">
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
              <button v-else class="image-frame" type="button" @click="openViewer(resultUrl)">
                <img :src="resultUrl" alt="增强结果预览" />
              </button>
            </article>
          </div>

          <div class="status-strip" :class="{ error: statusKind === 'error', success: statusKind === 'success' }">
            <span>{{ statusText }}</span>
            <span v-if="processing" class="pulse">模型推理中</span>
          </div>

          <section class="history-strip" aria-label="任务记录">
            <div class="history-title">最近任务</div>
            <button v-for="job in jobs" :key="job.id" class="history-item" type="button" @click="restoreJob(job)">
              <img :src="job.thumb" alt="" />
              <span>{{ job.modelTitle }}</span>
              <small>{{ job.elapsed }}</small>
            </button>
            <div v-if="jobs.length === 0" class="history-empty">完成一次超分后会在这里保留快捷记录</div>
          </section>
        </section>

        <aside class="inspector" aria-label="处理参数">
          <div class="panel-heading">
            <div>
              <h2>处理参数</h2>
              <p>模型会按需加载并缓存</p>
            </div>
            <button class="small-button" type="button" @click="refreshHealth">刷新</button>
          </div>

          <label class="field">
            <span>模型</span>
            <select v-model="settings.model_key" @change="syncModelDefaults">
              <option v-for="model in models" :key="model.key" :value="model.key">{{ model.title }}</option>
            </select>
          </label>

          <div class="model-note">
            <strong>{{ selectedModel?.title || "模型" }}</strong>
            <p>{{ selectedModel?.description || "加载模型信息中。" }}</p>
            <small>{{ selectedModel?.arch || "-" }} · 标称 {{ selectedModel?.scale || "-" }}x · {{ selectedModel?.supports_denoise ? "支持降噪" : "固定权重" }}</small>
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
          </div>

          <button class="run-button" type="button" :disabled="!selectedFile || processing" @click="runEnhance">
            <span>{{ processing ? "处理中..." : "开始超分" }}</span>
            <svg viewBox="0 0 24 24" fill="none"><path d="M5 12h14m0 0-5-5m5 5-5 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
          </button>
          <button class="clear-button" type="button" @click="clearAll">清空工作区</button>
        </aside>
      </div>

      <section v-else-if="activePage === 'models'" class="feature-page models-page">
        <div class="page-header">
          <div>
            <h2>模型管理</h2>
            <p>查看权重状态、模型特性，并快速切换到工作台。</p>
          </div>
          <button class="small-button" type="button" @click="refreshHealth">刷新状态</button>
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
              <button class="secondary-button" type="button" @click="chooseModel(model)">
                {{ model.key === settings.model_key ? "回到工作台" : "使用此模型" }}
              </button>
            </div>
          </article>
        </div>
      </section>

      <section v-else class="feature-page jobs-page">
        <div class="page-header">
          <div>
            <h2>任务记录</h2>
            <p>保留本次会话内完成的超分结果，方便回看和下载。</p>
          </div>
          <button class="small-button" type="button" :disabled="jobs.length === 0" @click="clearJobs">清空记录</button>
        </div>

        <div v-if="jobs.length === 0" class="empty-state">
          <strong>还没有任务</strong>
          <span>回到工作台上传图片并完成一次超分后，这里会出现结果记录。</span>
          <button class="secondary-button" type="button" @click="activePage = 'workspace'">去工作台</button>
        </div>

        <div v-else class="jobs-list">
          <article v-for="job in jobs" :key="job.id" class="job-row">
            <button class="job-thumb" type="button" @click="restoreJob(job)">
              <img :src="job.thumb" alt="" />
            </button>
            <div class="job-main">
              <h3>{{ job.modelTitle }}</h3>
              <p>{{ job.fileName }} · {{ job.createdAt }} · {{ job.elapsed }}</p>
              <span>{{ job.dimensions }}</span>
            </div>
            <div class="job-actions">
              <button class="secondary-button" type="button" @click="restoreJob(job)">查看结果</button>
              <a class="text-button" :href="job.url" download>下载</a>
            </div>
          </article>
        </div>
      </section>
    </section>

    <div v-if="viewerUrl" class="lightbox" role="dialog" aria-modal="true" @click.self="closeViewer">
      <div class="lightbox-toolbar">
        <span>{{ viewerTitle }}</span>
        <div>
          <button type="button" @click="setViewerZoom(zoom - 0.25)">-</button>
          <button type="button" @click="setViewerZoom(1)">{{ Math.round(zoom * 100) }}%</button>
          <button type="button" @click="setViewerZoom(zoom + 0.25)">+</button>
          <button type="button" @click="closeViewer">关闭</button>
        </div>
      </div>
      <div class="lightbox-canvas" title="滚轮缩放" @wheel="handleViewerWheel">
        <img :src="viewerUrl" :style="{ transform: `scale(${zoom})` }" alt="放大查看" />
      </div>
    </div>

    <input ref="fileInput" class="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" @change="handleFileInput" />
  </main>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from "vue";

const fileInput = ref(null);
const selectedFile = ref(null);
const previewUrl = ref("");
const resultUrl = ref("");
const resultMeta = ref("");
const statusText = ref("就绪。上传图片后设置参数并开始超分。");
const statusKind = ref("idle");
const processing = ref(false);
const health = ref(null);
const models = ref([]);
const jobs = ref([]);
const activePage = ref("workspace");
const viewMode = ref("compare");
const viewerUrl = ref("");
const viewerTitle = ref("图片查看");
const zoom = ref(1);

const pages = [
  {
    id: "workspace",
    short: "工作台",
    title: "Real-ESRGAN Studio",
    description: "本地图像超分工作台，保留 Real-ESRGAN 推理，换成 Vue 产品界面。",
  },
  {
    id: "models",
    short: "模型",
    title: "模型管理",
    description: "检查模型权重状态，选择合适的 Real-ESRGAN 模型。",
  },
  {
    id: "jobs",
    short: "任务",
    title: "任务记录",
    description: "查看本次会话的超分结果，快速回到输出图和下载文件。",
  },
];

const settings = reactive({
  model_key: "RealESRGAN_x4plus",
  outscale: 4,
  denoise_strength: 0.5,
  tile_size: 0,
  face_enhance: false,
  use_fp32: false,
  alpha_upsampler: "realesrgan",
});

const selectedModel = computed(() => models.value.find((model) => model.key === settings.model_key));
const pageMeta = computed(() => pages.find((page) => page.id === activePage.value) || pages[0]);
const activePreviewUrl = computed(() => {
  if (viewMode.value === "output") return resultUrl.value;
  return previewUrl.value || resultUrl.value;
});
const deviceShortLabel = computed(() => {
  const type = health.value?.device?.type;
  if (type === "mps") return "MPS";
  if (type === "cuda") return "CUDA";
  return "CPU";
});

function objectUrl(file) {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = URL.createObjectURL(file);
}

function acceptFile(file) {
  if (!file) return;
  selectedFile.value = file;
  objectUrl(file);
  resultUrl.value = "";
  resultMeta.value = "";
  statusText.value = "已载入原图。可以开始超分。";
  statusKind.value = "idle";
}

function handleFileInput(event) {
  acceptFile(event.target.files?.[0]);
}

function handleDrop(event) {
  acceptFile(event.dataTransfer.files?.[0]);
}

function syncModelDefaults() {
  if (selectedModel.value?.scale) {
    settings.outscale = selectedModel.value.scale;
  }
}

function chooseModel(model) {
  settings.model_key = model.key;
  syncModelDefaults();
  activePage.value = "workspace";
  statusKind.value = "idle";
  statusText.value = `已切换到 ${model.title}。可以继续上传或开始超分。`;
}

async function refreshHealth() {
  const [healthResponse, modelsResponse] = await Promise.all([fetch("/api/health"), fetch("/api/models")]);
  health.value = await healthResponse.json();
  models.value = await modelsResponse.json();
  if (!models.value.some((model) => model.key === settings.model_key) && models.value[0]) {
    settings.model_key = models.value[0].key;
    syncModelDefaults();
  }
}

async function runEnhance() {
  if (!selectedFile.value || processing.value) return;
  processing.value = true;
  statusKind.value = "idle";
  statusText.value = "正在准备模型并执行超分...";
  const form = new FormData();
  form.append("image", selectedFile.value);
  Object.entries(settings).forEach(([key, value]) => form.append(key, value));

  try {
    const response = await fetch("/api/enhance", {
      method: "POST",
      body: form,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || payload.message || "处理失败");
    }

    resultUrl.value = payload.url;
    resultMeta.value = `${payload.output.width} x ${payload.output.height}`;
    statusText.value = payload.status;
    statusKind.value = "success";
    viewMode.value = "compare";
    jobs.value.unshift({
      id: payload.id,
      thumb: payload.url,
      modelTitle: selectedModel.value?.title || settings.model_key,
      elapsed: `${payload.elapsed.toFixed(1)}s`,
      createdAt: new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }),
      dimensions: `${payload.input.width} x ${payload.input.height} -> ${payload.output.width} x ${payload.output.height}`,
      fileName: selectedFile.value?.name || "上传图片",
      url: payload.url,
    });
    jobs.value = jobs.value.slice(0, 6);
  } catch (error) {
    statusText.value = error.message;
    statusKind.value = "error";
  } finally {
    processing.value = false;
  }
}

function clearAll() {
  selectedFile.value = null;
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = "";
  resultUrl.value = "";
  resultMeta.value = "";
  statusKind.value = "idle";
  statusText.value = "已清空。拖入图片即可开始。";
  if (fileInput.value) fileInput.value.value = "";
}

function restoreJob(job) {
  resultUrl.value = job.url;
  resultMeta.value = job.dimensions || job.modelTitle;
  viewMode.value = "output";
  activePage.value = "workspace";
  statusKind.value = "success";
  statusText.value = `已打开任务结果: ${job.modelTitle}`;
}

function clearJobs() {
  jobs.value = [];
}

function setViewerZoom(value) {
  zoom.value = Math.min(6, Math.max(0.25, Number(value.toFixed(2))));
}

function handleViewerWheel(event) {
  event.preventDefault();
  const direction = event.deltaY < 0 ? 1 : -1;
  const step = event.altKey ? 0.1 : 0.2;
  setViewerZoom(zoom.value + direction * step);
}

function openViewer(url) {
  if (!url) return;
  viewerUrl.value = url;
  viewerTitle.value = url === resultUrl.value ? "增强结果" : "原图";
  zoom.value = 1;
}

function closeViewer() {
  viewerUrl.value = "";
}

onMounted(() => {
  refreshHealth().catch((error) => {
    statusKind.value = "error";
    statusText.value = `后端连接失败: ${error.message}`;
  });
});
</script>
