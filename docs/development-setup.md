# Development Setup Guide

## Prerequisites Installation

### 1. Python 3.11+ Setup

#### Windows
```powershell
# Download and install Python 3.11+ from python.org
# Or use winget
winget install Python.Python.3.11

# Verify installation
python --version
pip --version
```

#### macOS
```bash
# Using Homebrew
brew install python@3.11

# Or using pyenv
pyenv install 3.11.7
pyenv global 3.11.7
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-pip
```

### 2. Node.js 18+ Setup

#### Windows
```powershell
# Download from nodejs.org or use winget
winget install OpenJS.NodeJS

# Verify installation
node --version
npm --version
```

#### macOS
```bash
# Using Homebrew
brew install node@18

# Or using nvm
nvm install 18
nvm use 18
```

#### Linux
```bash
# Using NodeSource repository
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### 3. Docker Desktop Installation

#### Windows
1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Install with WSL 2 backend
3. Start Docker Desktop

#### macOS
```bash
# Using Homebrew
brew install --cask docker

# Or download from docker.com
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER
# Log out and back in
```

### 4. Azure CLI Installation

#### Windows
```powershell
# Using winget
winget install Microsoft.AzureCLI

# Or download MSI from Microsoft
```

#### macOS
```bash
# Using Homebrew
brew install azure-cli
```

#### Linux
```bash
# Ubuntu/Debian
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### 5. Azure Functions Core Tools

#### All Platforms
```bash
# Using npm (after Node.js installation)
npm install -g azure-functions-core-tools@4 --unsafe-perm true
```

## Project Setup

### 1. Repository Initialization

```bash
# Create new repository
mkdir fondastic
cd fondastic

# Initialize git
git init
git checkout -b main

# Create initial structure
mkdir -p {frontend,backend,functions,docs,tests,infrastructure}
mkdir -p backend/{app,tests}
mkdir -p frontend/{src,public}
mkdir -p functions/{background_removal,shared}
```

### 2. Environment Configuration

Create `.env` file in project root:
```env
# Azure Configuration
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
AZURE_TENANT_ID=your_tenant_id
AZURE_SUBSCRIPTION_ID=your_subscription_id

# Application Configuration
ENVIRONMENT=development
API_BASE_URL=http://localhost:8000
FUNCTIONS_BASE_URL=http://localhost:7071

# ML Configuration
MODEL_CONTAINER_NAME=models
DEFAULT_MODEL_TYPE=u2net
MAX_IMAGE_SIZE_MB=10
PROCESSING_TIMEOUT_SECONDS=30

# Development Configuration
DEBUG=true
LOG_LEVEL=debug
RELOAD=true
```

Create `.env.example` (without secrets):
```env
# Azure Configuration (required)
AZURE_STORAGE_ACCOUNT_NAME=
AZURE_CLIENT_ID=
AZURE_CLIENT_SECRET=
AZURE_TENANT_ID=
AZURE_SUBSCRIPTION_ID=

# Application Configuration
ENVIRONMENT=development
API_BASE_URL=http://localhost:8000
FUNCTIONS_BASE_URL=http://localhost:7071

# ML Configuration
MODEL_CONTAINER_NAME=models
DEFAULT_MODEL_TYPE=u2net
MAX_IMAGE_SIZE_MB=10
PROCESSING_TIMEOUT_SECONDS=30
```

### 3. Azure Resources Setup

#### Login to Azure
```bash
az login
az account show
```

#### Create Resource Group
```bash
az group create --name fondastic-dev-rg --location eastus
```

#### Create Storage Account
```bash
az storage account create \
  --name fondasticstorage \
  --resource-group fondastic-dev-rg \
  --location eastus \
  --sku Standard_LRS \
  --allow-blob-public-access false
```

#### Create Service Principal
```bash
az ad sp create-for-rbac \
  --name fondastic-dev-sp \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/fondastic-dev-rg \
  --json-auth
```

Save the output and use it to fill your `.env` file.

#### Create Blob Containers
```bash
# Get storage account key
STORAGE_KEY=$(az storage account keys list \
  --resource-group fondastic-dev-rg \
  --account-name fondasticstorage \
  --query '[0].value' \
  --output tsv)

# Create containers
az storage container create \
  --name models \
  --account-name fondasticstorage \
  --account-key $STORAGE_KEY

az storage container create \
  --name logs \
  --account-name fondasticstorage \
  --account-key $STORAGE_KEY
```

## Backend Development Setup

### 1. FastAPI Project Structure

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create project structure
mkdir -p app/{api,core,services,models,utils}
mkdir -p tests/{unit,integration}
```

### 2. Requirements File

Create `backend/requirements.txt`:
```txt
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Azure Services
azure-storage-blob==12.19.0
azure-identity==1.15.0
azure-functions==1.18.0

# ML and Image Processing
onnxruntime==1.16.3
pillow==10.1.0
opencv-python==4.8.1.78
numpy==1.24.3

# HTTP Client
httpx==0.25.2
aiohttp==3.9.1

# Utilities
python-dotenv==1.0.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.12.0
black==23.11.0
isort==5.12.0
flake8==6.1.0
mypy==1.7.1
```

### 3. FastAPI Application Structure

Create `backend/app/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.api import router as api_router
from app.core.config import settings

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

# Serve React static files in production
if not settings.DEBUG and os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    @app.get("/")
    async def serve_react_app():
        return FileResponse("static/index.html")
    
    @app.get("/{path:path}")
    async def serve_react_routes(path: str):
        if path.startswith("api/"):
            # Let FastAPI handle API routes
            return {"error": "Not found"}
        return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
```

### 4. Configuration Management

Create `backend/app/core/config.py`:
```python
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # Application
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Azure
    AZURE_STORAGE_ACCOUNT_NAME: str
    AZURE_CLIENT_ID: str
    AZURE_CLIENT_SECRET: str
    AZURE_TENANT_ID: str
    AZURE_SUBSCRIPTION_ID: str
    
    # Functions
    FUNCTIONS_BASE_URL: str = "http://localhost:7071"
    
    # ML Configuration
    MODEL_CONTAINER_NAME: str = "models"
    DEFAULT_MODEL_TYPE: str = "u2net"
    MAX_IMAGE_SIZE_MB: int = 10
    PROCESSING_TIMEOUT_SECONDS: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

### 5. API Routes Structure

Create `backend/app/api/__init__.py`:
```python
from fastapi import APIRouter

from .health import router as health_router
from .process import router as process_router
from .models import router as models_router

router = APIRouter()

router.include_router(health_router, prefix="/health", tags=["health"])
router.include_router(process_router, prefix="/process", tags=["processing"])
router.include_router(models_router, prefix="/models", tags=["models"])
```

## Frontend Development Setup

### 1. React Project Initialization

```bash
cd frontend

# Create React app with TypeScript
npx create-react-app . --template typescript

# Or with Vite (faster alternative)
npm create vite@latest . -- --template react-ts
```

### 2. Dependencies Installation

```bash
# UI Framework
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material @mui/lab

# HTTP Client
npm install axios react-query

# State Management
npm install zustand

# Form Handling
npm install react-hook-form @hookform/resolvers yup

# File Upload
npm install react-dropzone

# Utilities
npm install clsx date-fns

# Development Dependencies
npm install -D @types/react @types/react-dom
npm install -D eslint prettier husky lint-staged
npm install -D @testing-library/react @testing-library/jest-dom
npm install -D @testing-library/user-event
```

### 3. TypeScript Configuration

Update `frontend/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": [
      "dom",
      "dom.iterable",
      "ES6"
    ],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "baseUrl": "src",
    "paths": {
      "@/*": ["./*"],
      "@/components/*": ["components/*"],
      "@/services/*": ["services/*"],
      "@/types/*": ["types/*"],
      "@/utils/*": ["utils/*"],
      "@/hooks/*": ["hooks/*"]
    }
  },
  "include": [
    "src"
  ]
}
```

### 4. Project Structure

```bash
cd frontend/src

# Create folder structure
mkdir -p {components,pages,services,hooks,types,utils,styles}
mkdir -p components/{common,layout,ui}
```

## Azure Functions Setup

### 1. Functions Project Initialization

```bash
cd functions

# Initialize Functions project
func init . --python

# Create background removal function
func new --name background_removal --template "HTTP trigger" --authlevel anonymous
```

### 2. Requirements File

Create `functions/requirements.txt`:
```txt
azure-functions==1.18.0
azure-storage-blob==12.19.0
azure-identity==1.15.0

# ML Dependencies
onnxruntime==1.16.3
pillow==10.1.0
opencv-python-headless==4.8.1.78
numpy==1.24.3

# Utilities
python-dotenv==1.0.0
```

### 3. Function Configuration

Update `functions/host.json`:
```json
{
  "version": "2.0",
  "logging": {
    "applicationInsights": {
      "samplingSettings": {
        "isEnabled": true,
        "excludedTypes": "Request"
      }
    }
  },
  "extensionBundle": {
    "id": "Microsoft.Azure.Functions.ExtensionBundle",
    "version": "[4.*, 5.0.0)"
  },
  "functionTimeout": "00:02:00",
  "healthMonitor": {
    "enabled": true,
    "healthCheckInterval": "00:00:30",
    "healthCheckWindow": "00:02:00",
    "healthCheckThreshold": 6,
    "counterThreshold": 0.80
  }
}
```

## Docker Development Setup

### 1. Backend Dockerfile

Create `backend/Dockerfile`:
```dockerfile
FROM python:3.11-slim

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

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose for Development

Create `docker-compose.yml` in project root:
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DEBUG=true
      - ENVIRONMENT=development
      - FUNCTIONS_BASE_URL=http://functions:7071
    env_file:
      - .env
    volumes:
      - ./backend:/app
    depends_on:
      - azurite

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    volumes:
      - ./frontend:/app
      - /app/node_modules

  functions:
    build: ./functions
    ports:
      - "7071:7071"
    environment:
      - AzureWebJobsStorage=UseDevelopmentStorage=true
    env_file:
      - .env
    volumes:
      - ./functions:/home/site/wwwroot
    depends_on:
      - azurite

  azurite:
    image: mcr.microsoft.com/azure-storage/azurite
    ports:
      - "10000:10000"
      - "10001:10001"
      - "10002:10002"
    volumes:
      - azurite_data:/data

volumes:
  azurite_data:
```

## Development Workflow

### 1. Daily Development Commands

```bash
# Start all services
docker-compose up --build

# Start individual services
cd backend && uvicorn app.main:app --reload
cd frontend && npm start
cd functions && func start

# Run tests
cd backend && pytest
cd frontend && npm test
cd functions && python -m pytest

# Code formatting
cd backend && black . && isort .
cd frontend && npm run format
```

### 2. Git Hooks Setup

Create `.husky/pre-commit`:
```bash
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

# Run linting and formatting
cd backend && black --check . && isort --check-only .
cd frontend && npm run lint && npm run type-check

# Run tests
cd backend && pytest tests/
cd frontend && npm test -- --watchAll=false --coverage
```

### 3. Environment Variables Validation

Create `scripts/validate-env.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()

required_vars = [
    'AZURE_STORAGE_ACCOUNT_NAME',
    'AZURE_CLIENT_ID',
    'AZURE_CLIENT_SECRET',
    'AZURE_TENANT_ID',
    'AZURE_SUBSCRIPTION_ID',
]

missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print(f"Missing required environment variables: {missing_vars}")
    exit(1)
else:
    print("All required environment variables are set!")
```

## IDE Configuration

### Visual Studio Code Settings

Create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "./backend/venv/bin/python",
  "typescript.preferences.importModuleSpecifier": "relative",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.associations": {
    "*.env": "dotenv"
  },
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black"
}
```

### Recommended Extensions

Create `.vscode/extensions.json`:
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-vscode.vscode-typescript-next",
    "ms-azuretools.vscode-azurefunctions",
    "ms-vscode.azurecli",
    "bradlc.vscode-tailwindcss",
    "esbenp.prettier-vscode",
    "ms-python.black-formatter",
    "ms-python.isort"
  ]
}
```

## Testing Setup

### Backend Testing

Create `backend/pytest.ini`:
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --cov=app
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=80
```

### Frontend Testing

Update `frontend/package.json` test script:
```json
{
  "scripts": {
    "test": "react-scripts test --coverage --watchAll=false",
    "test:watch": "react-scripts test",
    "test:coverage": "react-scripts test --coverage --watchAll=false --coverageDirectory=coverage"
  }
}
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Kill processes using required ports
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

2. **Azure authentication**: Verify service principal permissions
```bash
az role assignment list --assignee {client-id}
```

3. **ONNX runtime issues**: Install correct version for your platform
```bash
pip install onnxruntime-gpu  # For GPU support
pip install onnxruntime      # For CPU only
```

4. **Docker build failures**: Clear Docker cache
```bash
docker system prune -a
docker-compose build --no-cache
```

### Development Tips

- Use Azure Storage Explorer for blob management
- Monitor Functions logs in real-time: `func logs tail`
- Use React Developer Tools for frontend debugging
- Set up Azure Application Insights for production monitoring

This setup provides a complete development environment for building Fondastic from scratch. 