<template>
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
        @click="$emit('navigate', 'workspace')"
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
        @click="$emit('navigate', 'models')"
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
        @click="$emit('navigate', 'jobs')"
      >
        <svg viewBox="0 0 24 24" fill="none"><path d="M8 7h10M8 12h10M8 17h6M4 7h.01M4 12h.01M4 17h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round" /></svg>
        <span class="visually-hidden">任务记录</span>
      </button>
    </nav>
    <div class="sidebar-status" :class="deviceClass" title="推理设备">
      {{ deviceLabel }}
    </div>
  </aside>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  activePage: { type: String, required: true },
  deviceLabel: { type: String, default: "CPU" },
  deviceType: { type: String, default: "cpu" },
});

defineEmits(["navigate"]);

const deviceClass = computed(() => props.deviceType || "cpu");
</script>
