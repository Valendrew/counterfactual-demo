<script setup lang="ts">
import type { SmartphoneInference } from '@/types/api'
import { defineProps } from 'vue'

const props = defineProps<{
    inferenceResult: SmartphoneInference
}>()
</script>

<template>
    <div class="flex flex-col">
        <button
            class="w-full h-10 rounded-md bg-secondary p-2 text-white"
            @click="$emit('fetchInference')"
        >
            Compute prediction
        </button>
        <div v-if="Object.keys(inferenceResult).length > 0">
            <p v-if="inferenceResult.ground_truth !== inferenceResult.prediction">
                <span class="text-red-500">Incorrect prediction</span>
            </p>
            <p>Minimum price range: {{ inferenceResult.min_price }}</p>
            <p>Maximum price range: {{ inferenceResult.max_price }}</p>
        </div>
    </div>
</template>
