# API Specification

## Overview

Fondastic provides RESTful API endpoints for image processing, status monitoring, and model management. The API is built with FastAPI and provides automatic OpenAPI documentation.

## Base URLs

- **Development**: `http://localhost:8000`
- **Production**: `https://fondastic.azurecontainerapps.io`

## Authentication

Currently, the API does not require authentication. Future versions will implement:
- JWT token authentication
- API key authentication for programmatic access
- Rate limiting per user/IP

## Content Types

- **Request**: `multipart/form-data` for file uploads, `application/json` for other requests
- **Response**: `application/json` for all responses

## Error Handling

All errors follow consistent format:
```json
{
  "error": true,
  "message": "Human-readable error message",
  "code": "ERROR_CODE",
  "details": {
    "field": "Additional error details"
  }
}
```

## Endpoints

### Health Check

#### `GET /api/health`

Check API health status and dependencies.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "functions": "healthy",
    "storage": "healthy"
  }
}
```

**Status Codes:**
- `200` - All services healthy
- `503` - One or more services unhealthy

---

### Process Image

#### `POST /api/process-image`

Process an image to remove background using AI models.

**Request:**
```
Content-Type: multipart/form-data

Fields:
- image: File (required) - Image file to process
- model: String (optional) - Model type ("u2net", "isnet", "u2netp")
- quality: String (optional) - Processing quality ("fast", "balanced", "high")
```

**Response:**
```json
{
  "success": true,
  "processing_time": 12.3,
  "image_metadata": {
    "original_size": {
      "width": 1920,
      "height": 1080
    },
    "processed_size": {
      "width": 1920,
      "height": 1080
    },
    "format": "PNG",
    "model_used": "u2net"
  },
  "result_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Error Examples:**
```json
{
  "error": true,
  "message": "File size exceeds maximum limit",
  "code": "FILE_TOO_LARGE",
  "details": {
    "max_size_mb": 10,
    "received_size_mb": 15.2
  }
}
```

**Status Codes:**
- `200` - Processing successful
- `400` - Invalid request (bad file, format, etc.)
- `413` - File too large
- `429` - Rate limit exceeded
- `500` - Internal server error
- `503` - ML service unavailable

---

### List Models

#### `GET /api/models`

Get list of available AI models and their specifications.

**Response:**
```json
{
  "models": [
    {
      "id": "u2net",
      "name": "U2-Net",
      "description": "Balanced accuracy and speed",
      "input_size": [320, 320],
      "processing_time_avg": 8.5,
      "accuracy_score": 0.92,
      "recommended_for": ["portraits", "objects"]
    },
    {
      "id": "isnet",
      "name": "IS-Net",
      "description": "High accuracy for complex images",
      "input_size": [1024, 1024],
      "processing_time_avg": 15.2,
      "accuracy_score": 0.96,
      "recommended_for": ["complex_scenes", "fine_details"]
    }
  ],
  "default_model": "u2net"
}
```

---

### Process Status (Future)

#### `GET /api/status/{job_id}`

Check processing status for async operations.

**Response:**
```json
{
  "job_id": "12345678-1234-5678-9012-123456789012",
  "status": "processing",
  "progress": 0.65,
  "estimated_time_remaining": 8.2,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:15Z"
}
```

**Status Values:**
- `queued` - Job in queue
- `processing` - Currently processing
- `completed` - Processing finished
- `failed` - Processing failed
- `expired` - Job expired

---

## Request/Response Examples

### Successful Image Processing

**Request:**
```bash
curl -X POST "http://localhost:8000/api/process-image" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/image.jpg" \
  -F "model=u2net"
```

**Response:**
```json
{
  "success": true,
  "processing_time": 8.7,
  "image_metadata": {
    "original_size": {"width": 800, "height": 600},
    "processed_size": {"width": 800, "height": 600},
    "format": "PNG",
    "file_size_kb": 245.8,
    "model_used": "u2net"
  },
  "result_image": "data:image/png;base64,iVBORw0..."
}
```

### Error Response

**Request with invalid file:**
```bash
curl -X POST "http://localhost:8000/api/process-image" \
  -F "image=@/path/to/document.pdf"
```

**Response:**
```json
{
  "error": true,
  "message": "Invalid file format. Supported formats: PNG, JPG, JPEG, BMP, TIFF",
  "code": "INVALID_FILE_FORMAT",
  "details": {
    "received_format": "application/pdf",
    "supported_formats": ["image/png", "image/jpeg", "image/bmp", "image/tiff"]
  }
}
```

## Rate Limiting

Current limits (will be implemented):
- **Anonymous users**: 10 requests per minute
- **Authenticated users**: 100 requests per minute
- **Premium users**: 1000 requests per minute

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

## File Constraints

- **Maximum file size**: 10 MB
- **Supported formats**: PNG, JPG, JPEG, BMP, TIFF
- **Maximum dimensions**: 4096 x 4096 pixels
- **Minimum dimensions**: 64 x 64 pixels

## SDK Examples

### Python
```python
import requests

def remove_background(image_path, model="u2net"):
    url = "http://localhost:8000/api/process-image"
    files = {"image": open(image_path, "rb")}
    data = {"model": model}
    
    response = requests.post(url, files=files, data=data)
    return response.json()
```

### JavaScript
```javascript
async function removeBackground(imageFile, model = "u2net") {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("model", model);
  
  const response = await fetch("/api/process-image", {
    method: "POST",
    body: formData
  });
  
  return await response.json();
}
```

### cURL
```bash
# Basic usage
curl -X POST "http://localhost:8000/api/process-image" \
  -F "image=@image.jpg" \
  -F "model=u2net"

# With quality setting
curl -X POST "http://localhost:8000/api/process-image" \
  -F "image=@image.jpg" \
  -F "model=isnet" \
  -F "quality=high"
```

## WebSocket Support (Future)

For real-time processing updates:

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/process");

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log("Progress:", data.progress);
};
```

## Monitoring Endpoints

### `GET /api/metrics`
Prometheus-compatible metrics endpoint.

### `GET /api/logs`
Recent application logs (admin only).

## OpenAPI Documentation

Interactive API documentation available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json` 