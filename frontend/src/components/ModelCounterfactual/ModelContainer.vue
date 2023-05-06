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
const counterfactualResult = ref({} as Smartphone)

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
    
    const res = await fetch('http://127.0.0.1:8000/counterfactual', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model_id: props.id,
            target: "lower"
        })
    })
    const results = await res.json()
    counterfactualResult.value = results
}
</script>

<template>
    <div
        id="modelCounterfactual"
        class="col-span-3 row-span-5 h-full w-full overflow-y-auto rounded-lg bg-primary"
    >
        <div class="flex flex-col w-full items-center gap-y-4 border-t-2 p-2">
            <Inference @fetch-inference="fetchInference" :inference-result="inferenceResult" />
            <Counterfacutal @fetch-counterfactual="fetchCounterfactual" :counterfactual-result="counterfactualResult"/>
        </div>
    </div>
</template>
