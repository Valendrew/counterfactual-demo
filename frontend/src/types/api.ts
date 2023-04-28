export type Smartphone = {
    misc_price: number | string;
    display_size: number;
    battery: number;
    memory_ram_gb: number;
    memory_rom_gb: number;
    main_camera_resolution: number;
    selfie_camera_resolution: number;
    has_oled_display: boolean;
    has_memory_card_slot: boolean;
    has_stereo_speakers: boolean;
    'has_3.5mm_jack': boolean;
    has_wlan_5ghz: boolean;
    has_nfc: boolean;
    has_wireless_charging: boolean;
    is_waterproof: boolean;
    network_technology: string;
    num_main_camera: number;
    launch_announced?: number;
    oem_model?: string;
    display_width?: number;
    display_height?: number;
    display_resolution?: string;
};

export type SmartphoneResponse = {
    [key: string]: Smartphone;
};

export type SmartphoneInference = {
    prediction: number;
    ground_truth: number;
    min_price: number;
    max_price: number;
}