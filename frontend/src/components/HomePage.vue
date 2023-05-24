<script setup lang="ts">
import InputSearch from './InputSearch.vue'
import FeaturesContainer from './ModelFeatures/FeaturesContainer.vue'
import ModelContainer from './ModelCounterfactual/ModelContainer.vue'
import type { Smartphone, SmartphoneResponse, SmartphoneHighlight } from '../types/api'
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
    searchList.value = results
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

        <!-- ORIGINAL SMARTPHONE CONTAINER -->
        <div class="order-last col-span-2 col-start-1 row-span-4 w-full h-full overflow-y-auto rounded-lg bg-primary">
            <div class="flex h-full flex-col">
                <div class="bg-secondary py-2">
                    <p class="text-center text-xl text-primary">ORIGINAL SMARTPHONE</p>
                </div>
                <FeaturesContainer
                    :smartphone="selectedModel"
                    :to-check="false"
                    :highlight-features="{} as SmartphoneHighlight"
                    v-if="Object.keys(selectedModel).length > 0"
                />
                <div v-else class="flex flex-col items-center pt-8">
                    <p class="text-md">No smartphone selected.</p>
                    <p class="text-sm">Please search one using the input above.</p>
                </div>
            </div>
        </div>

        <!-- COUNTERFACTUAL CONTAINER -->
        <div
            class="row-span-5 col-span-3 h-full w-full overflow-y-auto rounded-lg bg-primary"
        >
            <div class="flex h-full flex-col">
                <div class="bg-secondary py-2">
                    <p class="text-center text-xl text-primary">COUNTERFACTUAL SMARTPHONE</p>
                </div>
                <ModelContainer
                    :key="selectedIdx"
                    :smartphone="selectedModel"
                    :id="selectedIdx"
                    v-if="Object.keys(selectedModel).length > 0"
                />
                <div v-else class="flex flex-col items-center pt-8">
                    <p class="text-md">No smartphone selected.</p>
                    <p class="text-sm">Please search one using the input above.</p>
                </div>
            </div>
        </div>
    </div>
</template>
