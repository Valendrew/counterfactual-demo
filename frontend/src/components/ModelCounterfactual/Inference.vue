<script setup lang="ts">
import type { SmartphoneInference } from '@/types/api'
import { defineProps } from 'vue'

const props = defineProps<{
    inferenceResult: SmartphoneInference
}>()
</script>

<template>
    <div class="flex w-full flex-col gap-y-2">
        <button
            class="h-10 w-full rounded-md bg-secondary p-2 text-white"
            @click="$emit('fetchInference')"
        >
            Compute prediction
        </button>
        <div
            v-if="Object.keys(inferenceResult).length > 0"
            class="flex w-full flex-col justify-center"
        >
            <div class="flex flex-row justify-center">
                <p v-if="inferenceResult.ground_truth !== inferenceResult.prediction">
                    <span class="text-red-500">Incorrect prediction</span>
                </p>
            </div>

            <div class="flex flex-col justify-center gap-x-4 md:flex-row">
                <p>Minimum price range: {{ inferenceResult.min_price }}</p>
                <p class="hidden md:block">|</p>
                <p>Maximum price range: {{ inferenceResult.max_price }}</p>
            </div>
        </div>
    </div>
</template>
