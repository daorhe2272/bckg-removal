# Implementation Examples & Code Templates

This document provides concrete code examples and templates that can be used as starting points for implementing Fondastic.

## Azure Functions Implementation

### Background Removal Function

`functions/background_removal/__init__.py`:
```python
import azure.functions as func
import logging
import json
import base64
import io
import os
from typing import Optional
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# Global model cache
model_cache = {}

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function for background removal using ONNX models.
    """
    logging.info('Background removal function triggered.')
    
    try:
        # Parse request
        req_body = req.get_json()
        if not req_body:
            return func.HttpResponse(
                json.dumps({"error": "Invalid request body"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Extract parameters
        image_data = req_body.get('image_data')  # Base64 encoded
        model_type = req_body.get('model_type', 'u2net')
        
        if not image_data:
            return func.HttpResponse(
                json.dumps({"error": "No image data provided"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Decode image
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"error": f"Invalid image data: {str(e)}"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Load model
        session = load_model(model_type)
        if not session:
            return func.HttpResponse(
                json.dumps({"error": f"Failed to load model: {model_type}"}),
                status_code=500,
                mimetype="application/json"
            )
        
        # Process image
        result_image = process_image(image, session, model_type)
        
        # Convert result to base64
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format='PNG')
        result_base64 = base64.b64encode(output_buffer.getvalue()).decode()
        
        # Return result
        return func.HttpResponse(
            json.dumps({
                "success": True,
                "result_image": f"data:image/png;base64,{result_base64}",
                "model_used": model_type
            }),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Processing failed: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )

def load_model(model_type: str) -> Optional[ort.InferenceSession]:
    """Load ONNX model with caching."""
    if model_type in model_cache:
        return model_cache[model_type]
    
    try:
        # Download model from Azure Blob Storage
        model_path = download_model(model_type)
        if not model_path:
            return None
        
        # Load ONNX session
        session = ort.InferenceSession(model_path)
        model_cache[model_type] = session
        logging.info(f"Model {model_type} loaded successfully")
        return session
        
    except Exception as e:
        logging.error(f"Failed to load model {model_type}: {str(e)}")
        return None

def download_model(model_type: str) -> Optional[str]:
    """Download model from Azure Blob Storage."""
    try:
        # Azure credentials from environment
        account_name = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
        credential = DefaultAzureCredential()
        
        # Create blob client
        blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=credential
        )
        
        # Model filename mapping
        model_files = {
            'u2net': 'u2net.onnx',
            'isnet': 'isnet.onnx',
            'u2netp': 'u2netp.onnx'
        }
        
        model_filename = model_files.get(model_type)
        if not model_filename:
            return None
        
        # Download blob
        blob_path = f"production/{model_filename}"
        local_path = f"/tmp/{model_filename}"
        
        blob_client = blob_service_client.get_blob_client(
            container="models", 
            blob=blob_path
        )
        
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        return local_path
        
    except Exception as e:
        logging.error(f"Failed to download model {model_type}: {str(e)}")
        return None

def process_image(image: Image.Image, session: ort.InferenceSession, model_type: str) -> Image.Image:
    """Process image to remove background."""
    # Get model specifications
    input_size = get_model_input_size(model_type)
    mean, std = get_model_normalization(model_type)
    
    # Preprocess
    original_size = image.size
    img = image.convert('RGB')
    img = img.resize(input_size, Image.Resampling.BILINEAR)
    
    # Convert to array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Apply normalization
    for i in range(3):
        img_array[:, :, i] = (img_array[:, :, i] - mean[i]) / std[i]
    
    # Reshape for model input
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    # Run inference
    inputs = {session.get_inputs()[0].name: img_array}
    outputs = session.run(None, inputs)
    
    # Process output
    if model_type == 'isnet':
        mask = outputs[0][0] if isinstance(outputs[0], list) else outputs[0]
    else:
        mask = outputs[0]
    
    # Postprocess mask
    mask = mask.squeeze()
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, original_size)
    
    # Apply mask to original image
    img_array = np.array(image.convert('RGBA'))
    mask_normalized = mask.astype(np.float32) / 255.0
    img_array[:, :, 3] = (mask_normalized * 255).astype(np.uint8)
    
    return Image.fromarray(img_array, 'RGBA')

def get_model_input_size(model_type: str) -> tuple:
    """Get required input size for model."""
    sizes = {
        'u2net': (320, 320),
        'u2netp': (320, 320),
        'isnet': (1024, 1024)
    }
    return sizes.get(model_type, (320, 320))

def get_model_normalization(model_type: str) -> tuple:
    """Get normalization parameters for model."""
    if model_type == 'isnet':
        return ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    else:
        return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

## FastAPI Backend Implementation

### Main Application Structure

`backend/app/main.py`:
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.api import router as api_router
from app.core.config import settings
from app.core.logging import setup_logging

# Setup logging
setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Fondastic",
    description="Professional background removal using AI models",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Serve React static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    @app.get("/")
    async def serve_react_app():
        return FileResponse("static/index.html")
    
    @app.get("/{path:path}")
    async def catch_all(path: str):
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        return FileResponse("static/index.html")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    print(f"Starting Fondastic v{app.version}")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug mode: {settings.DEBUG}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down Fondastic")
```

### API Routes Implementation

`backend/app/api/process.py`:
```python
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import httpx
import base64
import time
from typing import Optional

from app.core.config import settings
from app.services.image_service import ImageService
from app.services.logging_service import LoggingService

router = APIRouter()

@router.post("/image")
async def process_image(
    image: UploadFile = File(...),
    model: str = Form("u2net"),
    quality: str = Form("balanced")
):
    """
    Process an image to remove background.
    
    Args:
        image: Image file to process
        model: Model type (u2net, isnet, u2netp)
        quality: Processing quality (fast, balanced, high)
    
    Returns:
        JSON response with processed image data
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image."
            )
        
        # Check file size
        content = await image.read()
        if len(content) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_IMAGE_SIZE_MB}MB"
            )
        
        # Validate model type
        valid_models = ['u2net', 'isnet', 'u2netp']
        if model not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Valid options: {valid_models}"
            )
        
        # Process image metadata
        image_service = ImageService()
        metadata = await image_service.extract_metadata(content, image.filename)
        
        # Encode image for Functions call
        image_base64 = base64.b64encode(content).decode()
        
        # Call Azure Functions
        async with httpx.AsyncClient(timeout=settings.PROCESSING_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{settings.FUNCTIONS_BASE_URL}/api/background_removal",
                json={
                    "image_data": image_base64,
                    "model_type": model
                }
            )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=503,
                detail="Image processing service unavailable"
            )
        
        result = response.json()
        processing_time = time.time() - start_time
        
        # Log the processing
        logging_service = LoggingService()
        await logging_service.log_prediction(
            metadata, processing_time, True, model
        )
        
        return JSONResponse({
            "success": True,
            "processing_time": processing_time,
            "image_metadata": metadata,
            "result_image": result["result_image"]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Log the error
        logging_service = LoggingService()
        await logging_service.log_prediction(
            metadata if 'metadata' in locals() else {},
            processing_time, False, model, str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image processing"
        )

@router.get("/status")
async def get_processing_status():
    """Get current processing service status."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(
                f"{settings.FUNCTIONS_BASE_URL}/api/health"
            )
        
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "functions_available": response.status_code == 200,
            "response_time": response.elapsed.total_seconds()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "functions_available": False,
            "error": str(e)
        }
```

### Services Implementation

`backend/app/services/image_service.py`:
```python
from PIL import Image
import io
from typing import Dict, Any

class ImageService:
    """Service for image processing and metadata extraction."""
    
    async def extract_metadata(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """Extract metadata from image."""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            return {
                "filename": filename,
                "format": image.format,
                "mode": image.mode,
                "size": {
                    "width": image.width,
                    "height": image.height
                },
                "file_size_bytes": len(image_data),
                "file_size_kb": round(len(image_data) / 1024, 2)
            }
        except Exception as e:
            return {
                "filename": filename,
                "error": f"Failed to extract metadata: {str(e)}"
            }
    
    def validate_image(self, image_data: bytes) -> bool:
        """Validate if data is a valid image."""
        try:
            Image.open(io.BytesIO(image_data))
            return True
        except Exception:
            return False
```

## React Frontend Implementation

### Main App Component

`frontend/src/App.tsx`:
```tsx
import React from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import { MainLayout } from './components/layout/MainLayout';
import { HomePage } from './pages/HomePage';

// Create Material-UI theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <MainLayout>
          <HomePage />
        </MainLayout>
        <ToastContainer
          position="bottom-right"
          autoClose={5000}
          hideProgressBar={false}
          newestOnTop={false}
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
        />
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
```

### Image Upload Component

`frontend/src/components/ImageUpload.tsx`:
```tsx
import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Paper,
  Typography,
  Button,
  LinearProgress,
  Alert,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import ImageIcon from '@mui/icons-material/Image';

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  isProcessing?: boolean;
  error?: string;
}

export const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageSelect,
  isProcessing = false,
  error,
}) => {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (file) {
        // Create preview
        const reader = new FileReader();
        reader.onload = () => {
          setPreview(reader.result as string);
        };
        reader.readAsDataURL(file);

        // Call parent handler
        onImageSelect(file);
      }
    },
    [onImageSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'],
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  return (
    <Box>
      <Paper
        {...getRootProps()}
        sx={{
          p: 4,
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.300',
          backgroundColor: isDragActive ? 'primary.50' : 'grey.50',
          cursor: 'pointer',
          textAlign: 'center',
          transition: 'all 0.2s ease',
          '&:hover': {
            borderColor: 'primary.main',
            backgroundColor: 'primary.50',
          },
        }}
      >
        <input {...getInputProps()} />
        
        {preview ? (
          <Box>
            <img
              src={preview}
              alt="Preview"
              style={{
                maxWidth: '100%',
                maxHeight: 300,
                borderRadius: 8,
              }}
            />
            <Typography variant="body2" sx={{ mt: 2 }}>
              Click or drag to replace image
            </Typography>
          </Box>
        ) : (
          <Box>
            <CloudUploadIcon sx={{ fontSize: 48, color: 'grey.400', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              {isDragActive ? 'Drop image here' : 'Upload an image'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Drag and drop an image here, or click to select
            </Typography>
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              Supported formats: PNG, JPG, JPEG, BMP, TIFF (max 10MB)
            </Typography>
          </Box>
        )}
      </Paper>

      {isProcessing && (
        <Box sx={{ mt: 2 }}>
          <LinearProgress />
          <Typography variant="body2" sx={{ mt: 1, textAlign: 'center' }}>
            Processing image...
          </Typography>
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
    </Box>
  );
};
```

### API Service

`frontend/src/services/api.ts`:
```typescript
import axios, { AxiosResponse } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: 30000,
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export interface ProcessImageResponse {
  success: boolean;
  processing_time: number;
  image_metadata: {
    filename: string;
    format: string;
    size: {
      width: number;
      height: number;
    };
    file_size_kb: number;
  };
  result_image: string;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
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

export const apiService = {
  // Health check
  async health(): Promise<HealthResponse> {
    const response: AxiosResponse<HealthResponse> = await api.get('/health');
    return response.data;
  },

  // Process image
  async processImage(
    file: File,
    model: string = 'u2net'
  ): Promise<ProcessImageResponse> {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('model', model);

    const response: AxiosResponse<ProcessImageResponse> = await api.post(
      '/process/image',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  },

  // Get available models
  async getModels(): Promise<ModelsResponse> {
    const response: AxiosResponse<ModelsResponse> = await api.get('/models');
    return response.data;
  },

  // Get processing status
  async getProcessingStatus(): Promise<any> {
    const response = await api.get('/process/status');
    return response.data;
  },
};
```

### Custom Hooks

`frontend/src/hooks/useImageProcessing.ts`:
```typescript
import { useMutation, useQueryClient } from 'react-query';
import { toast } from 'react-toastify';
import { apiService, ProcessImageResponse } from '../services/api';

interface ProcessImageParams {
  file: File;
  model?: string;
}

export const useImageProcessing = () => {
  const queryClient = useQueryClient();

  return useMutation<ProcessImageResponse, Error, ProcessImageParams>(
    async ({ file, model = 'u2net' }) => {
      return await apiService.processImage(file, model);
    },
    {
      onSuccess: (data) => {
        toast.success(
          `Image processed successfully in ${data.processing_time.toFixed(1)}s`
        );
        queryClient.invalidateQueries('processing-history');
      },
      onError: (error: any) => {
        const message = error.response?.data?.detail || 'Processing failed';
        toast.error(message);
      },
    }
  );
};
```

## Docker Configuration

### Production Dockerfile

`Dockerfile`:
```dockerfile
# Multi-stage build for production

# Stage 1: Build React frontend
FROM node:18-alpine as frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --only=production
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Python backend
FROM python:3.11-slim as backend-build
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application
COPY backend/ ./

# Copy frontend build from previous stage
COPY --from=frontend-build /app/frontend/build ./static

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### GitHub Actions Workflow

`.github/workflows/deploy.yml`:
```yaml
name: Build and Deploy

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
    
    - name: Install Python dependencies
      run: |
        cd backend
        pip install -r requirements.txt
    
    - name: Install Node.js dependencies
      run: |
        cd frontend
        npm ci
    
    - name: Run Python tests
      run: |
        cd backend
        pytest tests/ -v --cov=app
    
    - name: Run frontend tests
      run: |
        cd frontend
        npm test -- --coverage --watchAll=false
    
    - name: Lint code
      run: |
        cd backend && black --check . && isort --check-only .
        cd frontend && npm run lint

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    permissions:
      contents: read
      packages: write
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy-functions:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Azure Functions Core Tools
      run: npm install -g azure-functions-core-tools@4 --unsafe-perm true
    
    - name: Azure Login
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
    
    - name: Deploy to Azure Functions
      run: |
        cd functions
        func azure functionapp publish fondastic-functions --python

  deploy-container:
    needs: [build-and-push, deploy-functions]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Azure Login
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
    
    - name: Deploy to Azure Container Instances
      run: |
        az container create \
          --resource-group fondastic-rg \
          --name fondastic-prod \
          --image ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main \
          --cpu 2 --memory 4 \
          --ports 80 \
          --environment-variables \
            AZURE_STORAGE_ACCOUNT_NAME=${{ secrets.AZURE_STORAGE_ACCOUNT_NAME }} \
            FUNCTIONS_BASE_URL=https://fondastic-functions.azurewebsites.net \
            ENVIRONMENT=production \
          --restart-policy Always
```

This comprehensive set of examples provides concrete starting points for implementing each component of Fondastic. Each example includes error handling, logging, and follows best practices for production deployment. 