<script setup lang="ts">
import { defineProps } from 'vue'

const props = defineProps<{
    hasSpinner: boolean
}>()
</script>

<template>
    <div class="flex h-full w-full flex-col">
        <button
            class="w-full rounded-md bg-secondary p-2 text-white"
            @click="$emit('fetchCounterfactual')"
        >
            Compute counterfactual
        </button>
        <div
            v-if="props.hasSpinner"
            class="flex h-full w-full flex-row items-center justify-center"
        >
            <svg
                aria-hidden="true"
                class="mr-2 h-8 w-8 animate-spin fill-blue-600 text-gray-200 dark:text-gray-600 md:h-12 md:w-12 lg:h-16 lg:w-16"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
            >
                <g>
                    <circle cx="12" cy="2.5" r="1.5" opacity=".14" />
                    <circle cx="16.75" cy="3.77" r="1.5" opacity=".29" />
                    <circle cx="20.23" cy="7.25" r="1.5" opacity=".43" />
                    <circle cx="21.50" cy="12.00" r="1.5" opacity=".57" />
                    <circle cx="20.23" cy="16.75" r="1.5" opacity=".71" />
                    <circle cx="16.75" cy="20.23" r="1.5" opacity=".86" />
                    <circle cx="12" cy="21.5" r="1.5" />
                </g>
            </svg>
            <span class="sr-only">Loading...</span>
        </div>
    </div>
</template>
