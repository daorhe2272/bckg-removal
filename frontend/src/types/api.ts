// API Response Types
export interface ProcessImageResponse {
    success: boolean;
    processing_time: number;
    image_metadata: {
        filename: string;
        content_type: string;
        file_size_bytes: number;
        file_size_kb: number;
    };
    result_image: string; // base64 data URL
    model_used: string;
}

export interface HealthResponse {
    status: string;
    timestamp: string;
    version: string;
    environment: string;
    services: {
        functions: string;
        storage: string;
    };
}

export interface ModelInfo {
    id: string;
    name: string;
    description: string;
    input_size: [number, number];
    processing_time_avg: number;
    accuracy_score: number;
    recommended_for: string[];
}

export interface ModelsResponse {
    models: ModelInfo[];
    default_model: string;
}

export interface ProcessStatusResponse {
    status: 'healthy' | 'unhealthy';
    functions_available: boolean;
    response_time?: number;
    error?: string;
}

// Component Prop Types
export interface ImageUploadProps {
    onImageSelect: (file: File) => void;
    isProcessing?: boolean;
    error?: string;
    maxSizeBytes?: number;
}

export interface ModelSelectorProps {
    models: ModelInfo[];
    selectedModel: string;
    onModelChange: (modelId: string) => void;
    disabled?: boolean;
}

export interface ProcessingStatusProps {
    isProcessing: boolean;
    processingTime?: number;
    progress?: number;
    error?: string;
    success?: boolean;
}

export interface ImageComparisonProps {
    originalImage: string;
    processedImage?: string;
    isProcessing: boolean;
}

export interface DownloadButtonProps {
    imageData: string;
    filename: string;
    disabled?: boolean;
}

// Form Types
export interface ProcessImageFormData {
    image: File;
    model: string;
}

// Error Types
export interface ApiError {
    error: boolean;
    message: string;
    code?: string;
    details?: Record<string, unknown>;
}

// Constants
export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
export const SUPPORTED_FORMATS = [
    'image/png',
    'image/jpeg',
    'image/jpg',
    'image/bmp',
    'image/tiff'
] as const;

export const API_ENDPOINTS = {
    health: '/api/health',
    models: '/api/models',
    processImage: '/api/process/image',
    processStatus: '/api/process/status',
} as const; 