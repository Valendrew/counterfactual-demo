<script setup lang="ts">
import { defineProps } from 'vue'
import FeaturesContainer from '../ModelFeatures/FeaturesContainer.vue'
import type { Smartphone } from '@/types/api'

const props = defineProps<{
    counterfactualResult: Smartphone
}>()
</script>

<template>
    <div class="flex flex-col w-full">
        <button
            class="w-full rounded-md bg-secondary p-2 text-white"
            @click="$emit('fetchCounterfactual')"
        >
            Compute counterfactual
        </button>
        <div v-if="Object.keys(counterfactualResult).length > 0">
            <FeaturesContainer :smartphone="counterfactualResult" :to-check="false" :format-invert="true"/>
        </div>
    </div>
</template>
