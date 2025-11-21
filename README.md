# Fondastic

A modern, cost-efficient AI-powered background removal service built with Azure serverless architecture. Using FastAPI, React, and Azure Functions for optimal performance and cost optimization.

## 🎯 Project Overview

Fondastic is a professional background removal service optimized for low-volume usage (50 calls/day initially) with room to scale. The architecture uses Azure Functions for serverless ML inference, keeping costs minimal while maintaining high performance.

### Key Features
- ✨ **Modern Web Interface**: Professional React TypeScript frontend
- 🚀 **Serverless ML**: Azure Functions for cost-efficient ONNX model inference
- 📱 **Responsive Design**: Mobile-first interface with drag-and-drop upload
- 🎛️ **Multiple Models**: Support for U2-Net, IS-Net, and U2-NetP models
- 💰 **Cost Optimized**: ~$16-18/month total cost for low usage
- 🔄 **CI/CD Ready**: GitHub Actions deployment pipeline
- 📊 **Monitoring**: Azure Application Insights integration

## 🏗️ Architecture

```
┌─────────────────────────┐    ┌──────────────────────┐
│   FastAPI + React       │    │   Azure Functions    │
│   (Single Container)    │◄──►│   (ONNX Inference)   │
│   ~$15/month            │    │   ~$0-2/month        │
└─────────────────────────┘    └──────────────────────┘
              │                          │
              └──────────┬───────────────┘
                         ▼
               ┌──────────────────────┐
               │   Azure Blob Storage │
               │   (Models & Logs)    │
               │   ~$1/month          │
               └──────────────────────┘
```

## 🛠️ Technology Stack

### Frontend & API
- **FastAPI** - Modern Python web framework
- **React 18+** - TypeScript frontend with modern hooks
- **Material-UI** - Professional UI component library
- **React Query** - Server state management
- **Axios** - HTTP client with retry logic

### ML Inference
- **Azure Functions** - Serverless Python runtime
- **ONNX Runtime** - Optimized model inference
- **PIL/OpenCV** - Image processing pipeline
- **NumPy** - Numerical computations

### Infrastructure
- **Azure Container Instances** - Containerized FastAPI app
- **Azure Blob Storage** - Model and log storage
- **GitHub Actions** - CI/CD automation
- **Docker** - Containerization

## 📋 Prerequisites

### Development Environment
- **Python 3.11+** - Backend and Functions development
- **Node.js 18+** - Frontend development
- **Docker Desktop** - Containerization
- **Azure CLI** - Cloud resource management

### Azure Resources
- **Azure Subscription** - With sufficient credits/quota
- **Resource Group** - For organizing resources
- **Storage Account** - For blob storage
- **Service Principal** - For authentication

## 🚀 Quick Start Guide

### 1. Repository Setup
```bash
# Clone and initialize repository
git clone <repository-url>
cd ai-background-removal
git checkout -b development

# Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Azure Setup
```bash
# Login to Azure
az login

# Create resource group
az group create --name fondastic-rg --location eastus

# Create storage account
az storage account create \
  --name fondasticstorage \
  --resource-group fondastic-rg \
  --location eastus \
  --sku Standard_LRS

# Create service principal
az ad sp create-for-rbac --name fondastic-sp \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/fondastic-rg
```

### 3. Environment Configuration
Create `.env` file in project root:
```env
# Azure Configuration
AZURE_STORAGE_ACCOUNT_NAME=fondasticstorage
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
AZURE_TENANT_ID=your_tenant_id
AZURE_SUBSCRIPTION_ID=your_subscription_id

# Application Configuration
ENVIRONMENT=development
API_BASE_URL=http://localhost:8000
FUNCTIONS_BASE_URL=your_azure_functions_url

# ML Configuration
MODEL_CONTAINER_NAME=models
DEFAULT_MODEL_TYPE=u2net
MAX_IMAGE_SIZE_MB=10
PROCESSING_TIMEOUT_SECONDS=30
```

## 📁 Project Structure

```
/
├── frontend/                    # React TypeScript application
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API client services
│   │   ├── hooks/             # Custom React hooks
│   │   ├── types/             # TypeScript definitions
│   │   └── utils/             # Helper functions
│   ├── public/                # Static assets
│   └── package.json           # Dependencies
│
├── backend/                     # FastAPI application
│   ├── app/
│   │   ├── api/               # API route handlers
│   │   ├── core/              # Configuration
│   │   ├── services/          # Business logic
│   │   ├── models/            # Data models
│   │   └── utils/             # Helper functions
│   ├── tests/                 # Backend tests
│   ├── Dockerfile             # Container config
│   └── requirements.txt       # Dependencies
│
├── functions/                   # Azure Functions
│   ├── background_removal/     # Main ML function
│   ├── requirements.txt        # Function dependencies
│   ├── host.json              # Functions config
│   └── local.settings.json     # Local settings
│
├── .github/
│   └── workflows/              # CI/CD pipelines
│       ├── deploy.yml          # Main deployment
│       └── functions.yml       # Functions deployment
│
├── infrastructure/              # Infrastructure as Code
├── docs/                       # Documentation
├── tests/                      # Integration tests
├── docker-compose.yml          # Local development
├── OVERVIEW.md                 # Project overview
├── TASKS.md                   # Implementation tasks
└── README.md                  # This file
```

## 🔧 Development Workflow

### Local Development Setup

1. **Run locally on Windows (PowerShell) - 2 terminals**

   Terminal A — Backend (http://localhost:8000)
   ```powershell
   cd D:\fondastic\backend
   .\venv\Scripts\Activate.ps1
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   Terminal C — Frontend (http://localhost:5173)
   ```powershell
   cd D:\fondastic\frontend
   npm install
   $env:VITE_API_URL="http://localhost:8000"
   npm run dev
   ```

   CORS: Ensure backend allows Vite origin. In `backend/.env`, include:
   ```env
   ALLOWED_ORIGINS=["http://localhost:5173","http://127.0.0.1:5173"]
   ```

4. **Full Stack with Docker**
```bash
docker-compose up --build
# All services with hot reload
```

### Testing Strategy

```bash
# Backend tests
cd backend
pytest tests/ -v --cov=app

# Frontend tests
cd frontend
npm test -- --coverage

# Functions tests
cd functions
pytest tests/ -v

# Integration tests
cd tests
pytest integration/ -v

# End-to-end tests
npx playwright test
```

## 🚀 Deployment Guide

### GitHub Actions Setup

1. **Configure Repository Secrets**
   - `AZURE_CLIENT_ID`
   - `AZURE_CLIENT_SECRET`
   - `AZURE_TENANT_ID`
   - `AZURE_SUBSCRIPTION_ID`
   - `AZURE_STORAGE_ACCOUNT_NAME`

2. **Deployment Workflow**
   - Push to `dev` branch → Deploy to development
   - Push to `main` branch → Deploy to production
   - Pull requests → Run tests only

### Manual Deployment

```bash
# Build and push container
docker build -t fondastic-app .
az acr login --name fondasticsacr
docker tag fondastic-app fondasticsacr.azurecr.io/fondastic-app:latest
docker push fondasticsacr.azurecr.io/fondastic-app:latest

# Deploy Functions
cd functions
func azure functionapp publish fondastic-functions

# Deploy Container Instance
az container create \
  --resource-group fondastic-rg \
  --name fondastic-dev \
  --image fondasticsacr.azurecr.io/fondastic-app:latest \
  --cpu 2 --memory 4 \
  --ports 80 \
  --environment-variables \
    AZURE_STORAGE_ACCOUNT_NAME=fondasticstorage \
    FUNCTIONS_BASE_URL=https://fondastic-functions.azurewebsites.net
```

## 📊 Monitoring & Performance

### Key Metrics
- **Frontend Load Time**: <2 seconds
- **API Response Time**: <200ms (non-ML endpoints)
- **ML Processing Time**: <30 seconds per image
- **Monthly Cost**: <$20 for 50 calls/day
- **Uptime Target**: 99.9%

### Azure Monitoring
- **Application Insights** for performance tracking
- **Cost Management** for budget alerts
- **Function logs** for ML processing monitoring
- **Container metrics** for resource utilization

## 💰 Cost Analysis

### Monthly Cost Breakdown (50 calls/day)
```
Azure Container Instance (B1):     ~$15/month
Azure Functions (Consumption):     ~$0-2/month  
Azure Blob Storage:                 ~$1/month
Azure Application Insights:        ~$0-1/month
-------------------------------------------
Total:                             ~$16-19/month
```

### Scaling Cost Projections
```
500 calls/day:   ~$22/month
1000 calls/day:  ~$28/month
5000 calls/day:  ~$45/month
```

## 🔐 Security Best Practices

- **Input Validation**: File type, size, and content validation
- **Rate Limiting**: IP-based request throttling
- **Azure Security**: Service Principal with minimal permissions
- **CORS Configuration**: Restricted to known domains
- **Error Handling**: No sensitive information in error responses

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Follow code style**: Use ESLint, Prettier, and type checking
4. **Write tests**: Maintain >90% coverage
5. **Submit pull request**: With comprehensive description

### Code Style
- **Python**: Follow PEP 8, use Black formatter
- **TypeScript**: Use ESLint with strict rules
- **Commits**: Follow conventional commit format

## 📚 Additional Resources

- **[Project Overview](OVERVIEW.md)** - Detailed architecture and requirements
- **[Implementation Tasks](TASKS.md)** - Step-by-step development guide
- **[API Documentation](docs/api-docs.md)** - Complete API specification
- **[Deployment Guide](docs/deployment-guide.md)** - Production deployment
- **[Development Setup](docs/development-setup.md)** - Local development guide

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions, issues, or contributions:
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas
- **Documentation**: Check the `docs/` folder for detailed guides

---

**Note**: This README provides a comprehensive guide for building Fondastic from scratch. Follow the implementation tasks in `TASKS.md` for detailed step-by-step instructions. 