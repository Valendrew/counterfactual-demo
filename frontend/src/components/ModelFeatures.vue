<script setup lang="ts">
import { ref } from 'vue'
import { defineProps } from 'vue'
import type { Smartphone } from '@/types/api'

const props = defineProps<{
  smartphone: Smartphone
}>()

function expandModel(smartphone: Smartphone) {
  let features_structure = {
    Information: ['misc_price', 'launch_announced'],
    Display: ['display_size', 'has_oled_display', 'display_width', 'display_height'],
    Network: ['has_wlan_5ghz', 'network_technology'],
    Memory: ['has_memory_card_slot', 'memory_rom_gb', 'memory_ram_gb'],
    Audio: ['has_stereo_speakers', 'has_3.5mm_jack'],
    Camera: [
      'main_camera_resolution',
      'num_main_camera',
      'selfie_camera_resolution',
      'num_selfie_camera'
    ],
    Miscellaneous: [] as string[]
  }

  // get the features that have not been assigned to a category
  const leftover_features = Object.keys(smartphone).filter(
    (key) => !Object.values(features_structure).flat().includes(key)
  )
  features_structure = { ...features_structure, Miscellaneous: leftover_features }
  return features_structure
}

const smartphoneFeatures = ref(expandModel(props.smartphone))
</script>

<template>
  <div
    class="order-last col-span-2 col-start-1 row-span-4 flex h-full w-full flex-col items-center gap-y-2 overflow-y-auto rounded-lg bg-primary p-2"
  >
    <div>
      <p class="w-full text-xl">{{ smartphone.oem_model }}</p>
    </div>
    <div class="flex w-full flex-col gap-y-4 p-2">
      <div
        v-for="(features, category) in smartphoneFeatures"
        :key="category"
        class="flex w-full flex-col"
      >
        <div class="flex w-full flex-row">
          <p class="w-full bg-secondary text-center text-primary">{{ category }}</p>
        </div>
        <div
          v-for="feature in features"
          :key="feature"
          class="align-center flex flex-row justify-between"
        >
          <div class="align-center flex flex-row gap-x-2">
            <input type="checkbox" checked />
            <p>{{ feature }}</p>
          </div>
          <p>{{ smartphone[feature as keyof Smartphone] }}</p>
        </div>
      </div>
    </div>
  </div>
</template>
