<script setup lang="ts">
import InputSearch from './InputSearch.vue'
import ModelFeatures from './ModelFeatures.vue'
import ModelContainer from './ModelCounterfactual/ModelContainer.vue'
import type { Smartphone, SmartphoneResponse } from '../types/api'
import { ref } from 'vue'

const searchModel = ref('')
const searchList = ref({} as SmartphoneResponse)
const selectedModel = ref({} as Smartphone)
const selectedIdx = ref(-1)

async function searchInputModel(event: Event) {
    // Get the value from the event and set it to the input
    const inputValue = (event.target as HTMLInputElement).value
    searchModel.value = inputValue

    // Execute the search if the input is longer than 3 characters
    if (searchModel.value.length < 3) {
        searchList.value = {}
        return
    }

    // Post request to the server
    const res = await fetch('http://127.0.0.1:8000/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            name: searchModel.value
        })
    })

    // Get the response from the server
    const results = await res.json()
    searchList.value = JSON.parse(results)
}

function showModel(modelId: string) {
    const idxModel = parseInt(modelId)
    selectedModel.value = searchList.value[idxModel]
    selectedIdx.value = idxModel

    searchList.value = {}
    searchModel.value = ''
}
</script>

<template>
    <div
        class="grid h-screen w-screen grid-cols-5 grid-rows-5 gap-4 bg-background p-4 xl:px-24 xl:py-12"
    >
        <InputSearch
            :search-model="searchModel"
            :search-list="searchList"
            @search-input-model="searchInputModel"
            @show-model="showModel"
        />
        <ModelFeatures v-if="Object.keys(selectedModel).length > 0" :smartphone="selectedModel" />
        <ModelContainer
            v-if="Object.keys(selectedModel).length > 0"
            :key="selectedIdx"
            :smartphone="selectedModel"
            :id="selectedIdx"
        />
    </div>
</template>
