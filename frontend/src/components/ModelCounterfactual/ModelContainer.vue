<script setup lang="ts">
import { ref } from 'vue'
import { defineProps } from 'vue'
import type { Smartphone, SmartphoneInference } from '../../types/api'

import Counterfacutal from './Counterfactual.vue'
import Inference from './Inference.vue'

const props = defineProps<{
    id: number
    smartphone: Smartphone
}>()
const inferenceResult = ref({} as SmartphoneInference)

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
    console.log(results)
    inferenceResult.value = results
}
</script>

<template>
    <div
        id="modelCounterfactual"
        class="col-span-3 row-span-5 h-full w-full overflow-y-auto rounded-lg bg-primary"
    >
        <div class="grid-cols-2 grid w-full justify-center gap-x-8 border-t-2 p-2">
            <Counterfacutal />
            <Inference @fetch-inference="fetchInference" :inference-result="inferenceResult" />
        </div>
    </div>
</template>
