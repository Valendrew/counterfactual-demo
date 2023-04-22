<script setup lang="ts">
import { defineProps } from 'vue'

const props = defineProps<{
  modelName: string
}>()
</script>

<template>
  <div class="flex flex-col items-center">
    <button @click="$emit('showModel')">
      {{ props.modelName }}
    </button>
  </div>
</template>
