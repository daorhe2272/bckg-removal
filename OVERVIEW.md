# Project Overview: Fondastic

## 1. Project Vision & Goals

**Primary Goal**: Build Fondastic, a modern, cost-efficient AI-powered background removal service optimized for low-volume usage (50 calls/day initially) with room to scale.

**Key Objectives**:
- Modern web application with professional UI/UX
- Serverless ML inference for cost optimization
- CI/CD pipeline with dev/prod environments  
- Support for multiple ONNX models (U2-Net, IS-Net)
- Future-ready architecture for image editing, auth, and payments

## 2. Architecture Overview

### **Hybrid Serverless Architecture**
```
Frontend (React) + API (FastAPI) ──┐
     ↓ Single Container              │
Azure Container Instance             │
     (~$15/month)                    │
                                     │
     ↓ HTTP calls                    │
Azure Functions (Python)            │
     ↓ ONNX Runtime                  │
Background Removal ML                │
     (~$0-2/month for 50 calls/day)  │
                                     │
     ↓ Storage                       │
Azure Blob Storage                   │
     ↓ Models & Logs                 │
     (~$1/month)                     │
```

### **Cost Optimization**
- **Total Monthly Cost**: ~$16-18/month (vs $90+ for always-on GPU containers)
- **Serverless ML**: Pay only for actual inference time
- **Auto-scaling**: 0→1→0 instances based on demand
- **Free tier usage**: 1M Azure Functions executions/month free

## 3. Technology Stack

### **Frontend & API Layer**
- **Framework**: FastAPI (Python 3.11+)
- **Frontend**: React 18+ with TypeScript
- **UI Library**: Material-UI or Chakra UI for professional design
- **Styling**: CSS-in-JS or Tailwind CSS
- **State Management**: React Query for server state, Zustand for client state

### **ML Inference Layer**
- **Serverless**: Azure Functions (Python 3.11)
- **ML Runtime**: ONNX Runtime for optimized inference
- **Models**: U2-Net (320×320), IS-Net (1024×1024), U2-NetP
- **Image Processing**: PIL, OpenCV, NumPy

### **Storage & Data**
- **Model Storage**: Azure Blob Storage (production/ folder)
- **Logging**: Azure Blob Storage (JSON logs)
- **User Data**: Future - Azure Cosmos DB or PostgreSQL

### **DevOps & Deployment**
- **Containerization**: Docker for FastAPI app
- **CI/CD**: GitHub Actions
- **Hosting**: Azure Container Instances
- **Functions**: Azure Functions with Python runtime
- **Monitoring**: Azure Application Insights

## 4. Key System Requirements

### **4.1 Frontend Requirements**
- **Modern UI**: Professional, responsive design
- **File Upload**: Drag-and-drop with preview
- **Model Selection**: UI to choose between available models
- **Processing Status**: Real-time progress indicators
- **Download**: PNG with transparent background
- **Mobile Support**: Responsive design for all devices

### **4.2 API Requirements**
- **RESTful Design**: Clear, documented endpoints
- **File Handling**: Multipart form uploads
- **Error Handling**: Comprehensive error responses
- **Logging**: Request/response logging
- **CORS**: Frontend communication support
- **Health Checks**: Monitoring endpoints

### **4.3 ML Inference Requirements**
- **Multiple Models**: Support U2-Net, IS-Net, U2-NetP
- **Auto-scaling**: Start from 0, scale with demand
- **Model Loading**: Download from Blob Storage on cold start
- **Image Preprocessing**: Automatic resizing and normalization
- **Error Handling**: Graceful failure handling
- **Performance**: <30 second processing time

### **4.4 Future Requirements**
- **Authentication**: User accounts and sessions
- **Payment Integration**: Stripe/PayPal for subscriptions
- **Image Editing**: Basic editing tools (crop, resize, filters)
- **Batch Processing**: Multiple images at once
- **API Rate Limiting**: Usage quotas and throttling

## 5. Project Structure

```
/
├── frontend/                    # React TypeScript application
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API client services
│   │   ├── hooks/             # Custom React hooks
│   │   ├── types/             # TypeScript type definitions
│   │   └── utils/             # Helper functions
│   ├── public/                # Static assets
│   └── package.json           # Dependencies
│
├── backend/                     # FastAPI application
│   ├── app/
│   │   ├── api/               # API route handlers
│   │   ├── core/              # Configuration and settings
│   │   ├── services/          # Business logic services
│   │   ├── models/            # Data models
│   │   └── utils/             # Helper functions
│   ├── tests/                 # Backend tests
│   ├── Dockerfile             # Container configuration
│   └── requirements.txt       # Python dependencies
│
├── functions/                   # Azure Functions
│   ├── background_removal/     # Main inference function
│   ├── requirements.txt        # Function dependencies
│   ├── host.json              # Functions configuration
│   └── local.settings.json     # Local development settings
│
├── infrastructure/              # Infrastructure as Code
│   ├── azure-resources.json    # ARM templates
│   └── deploy-scripts/         # Deployment automation
│
├── .github/
│   └── workflows/              # CI/CD pipelines
│       ├── test-and-deploy.yml # Main pipeline
│       └── functions-deploy.yml # Functions deployment
│
├── docs/                       # Documentation
│   ├── api-docs.md            # API documentation
│   ├── deployment-guide.md     # Deployment instructions
│   └── development-setup.md    # Local development setup
│
├── tests/                      # Integration tests
├── docker-compose.yml          # Local development environment
├── OVERVIEW.md                 # This file
├── TASKS.md                   # Implementation tasks
└── README.md                  # Project documentation
```

## 6. Development Workflow

### **6.1 Local Development**
- **Frontend**: React dev server (localhost:3000)
- **Backend**: FastAPI with hot reload (localhost:8000)
- **Functions**: Azure Functions Core Tools (localhost:7071)
- **Storage**: Azurite emulator for local blob storage

### **6.2 Testing Strategy**
- **Frontend**: Jest + React Testing Library
- **Backend**: Pytest with async support
- **Functions**: Azure Functions testing framework
- **E2E**: Playwright for end-to-end testing
- **API**: Automated API testing with request validation

### **6.3 CI/CD Pipeline**
```
Push to branch → Run Tests → Build Images → Deploy
     ↓               ↓          ↓           ↓
   Lint Code    Unit Tests   Docker Build  Azure Deploy
   Type Check   E2E Tests    Functions     Environment
   Security     API Tests    Package       Validation
```

## 7. Environment Configuration

### **7.1 Development Environment**
- Local containers with hot reload
- Azure Functions emulator
- Mock authentication
- Test data and models

### **7.2 Production Environment**
- Azure Container Instances
- Azure Functions (consumption plan)
- Azure Blob Storage
- Production models and monitoring

### **7.3 Environment Variables**
```bash
# Azure Configuration
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
AZURE_TENANT_ID=your_tenant_id

# Application Configuration  
ENVIRONMENT=development|production
API_BASE_URL=http://localhost:8000
FUNCTIONS_BASE_URL=http://localhost:7071

# ML Configuration
MODEL_CONTAINER_NAME=models
DEFAULT_MODEL_TYPE=u2net
MAX_IMAGE_SIZE_MB=10
PROCESSING_TIMEOUT_SECONDS=30
```

## 8. Security & Performance

### **8.1 Security Requirements**
- Input validation and sanitization
- File type and size restrictions
- Rate limiting per IP
- CORS configuration
- Secure Azure credentials management

### **8.2 Performance Targets**
- **Frontend Load**: <2 seconds initial load
- **API Response**: <200ms for non-ML endpoints
- **ML Processing**: <30 seconds per image
- **Cold Start**: <10 seconds for Functions
- **Uptime**: 99.9% availability

## 9. Monitoring & Logging

### **9.1 Application Monitoring**
- Azure Application Insights integration
- Custom metrics for ML processing times
- Error tracking and alerting
- Performance monitoring

### **9.2 Business Metrics**
- Daily/monthly usage statistics
- Model performance metrics
- User engagement analytics
- Cost tracking and optimization

## 10. Future Roadmap

### **Phase 1** (Current): Core MVP
- Basic background removal
- Single container + serverless ML
- Dev/prod environments

### **Phase 2**: Enhanced Features
- User authentication
- Multiple image formats
- Basic image editing tools

### **Phase 3**: Commercial Features
- Payment integration
- Advanced editing capabilities
- Batch processing
- API access for developers

### **Phase 4**: Scale Optimization
- Microservices architecture
- CDN integration
- Advanced ML models
- Enterprise features

## 11. Success Metrics

### **Technical Metrics**
- 99.9% uptime
- <30s processing time
- <$50/month operational cost (first 6 months)
- Zero security incidents

### **Business Metrics**
- 50+ daily active users by month 3
- <5% error rate for image processing
- Positive user feedback (>4.5/5 rating)
- Clear path to monetization

## 12. Risk Mitigation

### **Technical Risks**
- **Cold start latency**: Model caching and warm-up strategies
- **Azure limits**: Monitor quotas and implement graceful degradation
- **Model accuracy**: Multiple model options and quality validation

### **Business Risks**
- **Cost overruns**: Automated cost alerts and usage monitoring
- **Scaling challenges**: Architecture designed for horizontal scaling
- **Competitive pressure**: Focus on user experience and unique features