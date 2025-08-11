import type { AxiosResponse } from 'axios';
import axios from 'axios';
import type {
    ApiError,
    HealthResponse,
    ModelsResponse,
    ProcessImageResponse,
    ProcessStatusResponse,
} from '../types/api';
import { API_ENDPOINTS } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000, // 30 seconds
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor
api.interceptors.request.use(
    (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
    },
    (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
    }
);

// Response interceptor
api.interceptors.response.use(
    (response) => {
        return response;
    },
    (error) => {
        console.error('API Response Error:', error.response?.data || error.message);

        // Transform error to our ApiError format
        const apiError: ApiError = {
            error: true,
            message: error.response?.data?.detail || error.message || 'An unexpected error occurred',
            code: error.response?.status?.toString(),
            details: error.response?.data,
        };

        return Promise.reject(apiError);
    }
);

export const apiService = {
    // Health check
    async health(): Promise<HealthResponse> {
        const response: AxiosResponse<HealthResponse> = await api.get(API_ENDPOINTS.health);
        return response.data;
    },

    // Get available models
    async getModels(): Promise<ModelsResponse> {
        const response: AxiosResponse<ModelsResponse> = await api.get(API_ENDPOINTS.models);
        return response.data;
    },

    // Process image for background removal
    async processImage(file: File, model: string = 'u2net'): Promise<ProcessImageResponse> {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('model', model);

        const response: AxiosResponse<ProcessImageResponse> = await api.post(
            API_ENDPOINTS.processImage,
            formData,
            {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                timeout: 60000, // Extended timeout for image processing
            }
        );
        return response.data;
    },

    // Get processing service status
    async getProcessingStatus(): Promise<ProcessStatusResponse> {
        const response: AxiosResponse<ProcessStatusResponse> = await api.get(API_ENDPOINTS.processStatus);
        return response.data;
    },
};

// Utility functions
export const downloadImage = (imageData: string, filename: string) => {
    const link = document.createElement('a');
    link.href = imageData;
    link.download = filename.replace(/\.[^/.]+$/, '') + '_no_background.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};

export const validateImageFile = (file: File): { isValid: boolean; error?: string } => {
    const MAX_SIZE = 10 * 1024 * 1024; // 10MB
    const SUPPORTED_TYPES = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];

    if (!SUPPORTED_TYPES.includes(file.type)) {
        return {
            isValid: false,
            error: `Unsupported file type. Supported formats: ${SUPPORTED_TYPES.join(', ')}`
        };
    }

    if (file.size > MAX_SIZE) {
        return {
            isValid: false,
            error: `File too large. Maximum size: ${(MAX_SIZE / 1024 / 1024).toFixed(1)}MB`
        };
    }

    return { isValid: true };
};

export default apiService; 