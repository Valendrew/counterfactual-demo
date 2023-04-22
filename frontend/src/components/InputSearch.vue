<script setup lang="ts">
import { defineProps } from 'vue'
import SearchList from './SearchList.vue'
import type { SmartphoneResponse } from '@/types/api'

const props = defineProps<{
  searchModel: String
  searchList: SmartphoneResponse
}>()

function showModel (modelId: string) {
  console.log(modelId)
}
</script>

<template>
  <div
    class="relative col-span-2 col-start-1 row-span-1 flex h-fit w-full flex-col items-center rounded-lg bg-primary"
  >
    <input
      class="rounded-x-lg h-12 w-full rounded-t-lg border bg-primary p-4 focus:border-secondary focus:outline-0"
      type="text"
      placeholder="Search smartphone..."
      :value="searchModel"
      @input="$emit('searchInputModel', $event)"
    />
    <div
      id="searchList"
      v-if="Object.keys(searchList).length > 0"
      class="rounded-x-lg absolute top-12 flex max-h-24 min-h-0 w-full flex-col gap-y-2 overflow-y-auto rounded-b-lg border border-secondary bg-primary"
    >
      <SearchList
        v-for="(value, key) in searchList"
        :key="key"
        :model-name="value.oem_model"
        @show-model="$emit('showModel', key)"
      />
    </div>
  </div>
</template>
