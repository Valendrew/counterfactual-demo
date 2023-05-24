<script setup lang="ts">
import { ref } from 'vue'
import { defineProps } from 'vue'
import type { Smartphone, SmartphoneInference, SmartphoneHighlight } from '../../types/api'

import FeaturesContainer from '../ModelFeatures/FeaturesContainer.vue'
import Counterfacutal from './Counterfactual.vue'
import Inference from './Inference.vue'

const props = defineProps<{
    id: number
    smartphone: Smartphone
}>()
const inferenceResult = ref({} as SmartphoneInference)
const counterfactualResult = ref({} as Smartphone)
const originalResult = ref({} as Smartphone)
const hasSpinner = ref(false)
const highlightFeatures = ref({} as SmartphoneHighlight)

async function fetchInference() {
    console.log(`fetching inference on id=${props.id.toString()}`)

    const res = await fetch('http://127.0.0.1:8000/inference', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model_id: props.id
        })
    })
    const results = await res.json()
    inferenceResult.value = results
}

async function fetchCounterfactual() {
    console.log(`fetching counterfactual on id=${props.id.toString()}`)
    if (Object.keys(inferenceResult.value).length === 0) {
        await fetchInference()
    }

    const targetCounterfactual =
        inferenceResult.value.ground_truth !== inferenceResult.value.prediction ? 'same' : 'lower'

    hasSpinner.value = true
    const res = await fetch('http://127.0.0.1:8000/counterfactual', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model_id: props.id,
            target: targetCounterfactual
        })
    })
    const results = await res.json()
    hasSpinner.value = false
    console.log(results)

    counterfactualResult.value = results.counterfactual
    originalResult.value = results.original

    Object.entries(originalResult.value).forEach(([key, value]) => {
        const cf_value = counterfactualResult.value[key as keyof Smartphone]
        if (cf_value !== value) {
            highlightFeatures.value[key as keyof SmartphoneHighlight] = true
        }
    })
}
</script>

<template>
    <div class="flex h-full w-full flex-col items-center gap-y-4 overflow-y-auto border-t-2 p-2">
        <div v-if="Object.keys(counterfactualResult).length > 0" class="w-full">
            <FeaturesContainer
                :smartphone="counterfactualResult"
                :to-check="false"
                :format-invert="true"
                :highlight-features="highlightFeatures"
            />
        </div>
        <div v-else class="flex w-full flex-col items-center">
            <Inference @fetch-inference="fetchInference" :inference-result="inferenceResult" />
            <Counterfacutal
                v-if="Object.keys(inferenceResult).length > 0"
                @fetch-counterfactual="fetchCounterfactual"
                :has-spinner="hasSpinner"
            />
        </div>
    </div>
</template>
