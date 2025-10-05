# 🚀 NASA Exoplanet API - Deployment Guide

**NASA Space Apps Challenge 2025 - Team BrainRot**

## 📋 **Quick Deployment Steps**

### **Option 1: Local Development (Recommended for Testing)**

#### **Step 1: Start Backend**
```bash
cd backend
pip install -r requirements.txt
python app.py
```
Backend will be available at: `http://localhost:5000`

#### **Step 2: Start Frontend**
```bash
cd frontend
npm install
npm start
```
Frontend will be available at: `http://localhost:3000`

---

### **Option 2: Docker Deployment (Recommended for Production)**

#### **Prerequisites**
- Docker installed
- Docker Compose installed

#### **Step 1: Build and Run**
```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

#### **Step 2: Verify Deployment**
- Backend: `http://localhost:5000`
- Frontend: `http://localhost:3000`
- Health Check: `http://localhost:5000/health`

---

### **Option 3: Cloud Deployment (Render.com)**

#### **Step 1: Prepare Repository**
```bash
# Ensure all files are committed
git add .
git commit -m "Production deployment"
git push origin main
```

#### **Step 2: Deploy to Render**
1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Render will automatically detect `render.yaml`
4. Click "Deploy"

#### **Step 3: Configure Environment**
- Backend will auto-deploy with `render.yaml` config
- Frontend will auto-deploy and connect to backend

---

## 🔧 **Detailed Setup Instructions**

### **Backend Setup**

#### **1. Install Dependencies**
```bash
cd backend
pip install -r requirements.txt
```

#### **2. Environment Variables (Optional)**
```bash
# Create .env file (optional)
FLASK_ENV=production
LOG_LEVEL=INFO
```

#### **3. Start Backend**
```bash
# Development
python app.py

# Production (with Gunicorn)
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

### **Frontend Setup**

#### **1. Install Dependencies**
```bash
cd frontend
npm install
```

#### **2. Environment Configuration**
```bash
# Create .env file (if needed)
REACT_APP_API_URL=http://localhost:5000/api
```

#### **3. Start Frontend**
```bash
# Development
npm start

# Production build
npm run build
npm install -g serve
serve -s build -l 3000
```

---

## 🐳 **Docker Deployment Details**

### **Using Docker Compose (Recommended)**

#### **1. Start Services**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### **2. Services Included**
- **API Backend**: Port 5000
- **React Frontend**: Port 3000
- **Redis Cache**: Port 6379 (optional)
- **Nginx Proxy**: Port 80 (optional)

### **Manual Docker Build**

#### **1. Build Backend Image**
```bash
docker build -t nasa-exoplanet-api .
```

#### **2. Run Backend Container**
```bash
docker run -d \
  --name nasa-api \
  -p 5000:5000 \
  nasa-exoplanet-api
```

#### **3. Build Frontend Image**
```bash
cd frontend
docker build -t nasa-exoplanet-frontend .
```

#### **4. Run Frontend Container**
```bash
docker run -d \
  --name nasa-frontend \
  -p 3000:3000 \
  -e REACT_APP_API_URL=http://localhost:5000/api \
  nasa-exoplanet-frontend
```

---

## ☁️ **Cloud Deployment Options**

### **Render.com (Recommended)**

#### **Automatic Deployment**
1. **Push to GitHub**: Ensure code is in GitHub repository
2. **Connect Render**: Link GitHub repo to Render
3. **Auto-Deploy**: Render uses `render.yaml` for configuration

#### **Manual Configuration**
- **Backend Service**: 
  - Build Command: `cd backend && pip install -r requirements.txt`
  - Start Command: `cd backend && gunicorn --bind 0.0.0.0:$PORT app:app`
- **Frontend Service**:
  - Build Command: `cd frontend && npm install && npm run build`
  - Start Command: `cd frontend && npm start`

### **Other Cloud Platforms**

#### **Heroku**
```bash
# Install Heroku CLI
# Create Heroku apps
heroku create nasa-exoplanet-api
heroku create nasa-exoplanet-frontend

# Deploy backend
git subtree push --prefix backend heroku main

# Deploy frontend
git subtree push --prefix frontend heroku main
```

#### **AWS/GCP/Azure**
- Use Docker images with container services
- Configure load balancers and auto-scaling
- Set up environment variables and secrets

---

## 🔍 **Verification Steps**

### **1. Health Checks**
```bash
# Backend health
curl http://localhost:5000/health

# API endpoints
curl http://localhost:5000/api/

# NASA pipeline status
curl http://localhost:5000/api/nasa/pipeline/status
```

### **2. Frontend Verification**
- Open `http://localhost:3000`
- Test file upload functionality
- Verify exoplanet analysis works
- Check results display correctly

### **3. Integration Testing**
```bash
# Test complete workflow
# 1. Upload a CSV file with exoplanet data
# 2. Run analysis
# 3. View results
# 4. Verify consistent results on repeat
```

---

## ⚡ **Performance Optimization**

### **Production Settings**

#### **Backend Optimization**
```bash
# Use Gunicorn with multiple workers
gunicorn --bind 0.0.0.0:5000 \
  --workers 4 \
  --threads 2 \
  --timeout 120 \
  --keep-alive 2 \
  app:app
```

#### **Frontend Optimization**
```bash
# Build optimized production bundle
npm run build

# Serve with compression
npm install -g serve
serve -s build -l 3000
```

### **Scaling Configuration**
- **CPU**: 2+ cores recommended
- **Memory**: 1GB+ recommended
- **Storage**: 5GB+ for data and logs
- **Network**: CDN for frontend assets

---

## 🛡️ **Security Configuration**

### **Environment Variables**
```bash
# Backend
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
LOG_LEVEL=INFO

# Frontend
REACT_APP_API_URL=https://your-api-domain.com/api
NODE_ENV=production
```

### **CORS Configuration**
- Backend CORS is pre-configured for localhost
- Update `app.py` CORS origins for production domains

### **SSL/HTTPS**
- Use reverse proxy (Nginx) for SSL termination
- Configure SSL certificates
- Redirect HTTP to HTTPS

---

## 📊 **Monitoring & Logs**

### **Log Files**
- Backend logs: `backend/logs/`
- Application logs: `nasa_api.log`
- Error logs: `errors.log`
- Performance logs: `performance.log`

### **Health Monitoring**
```bash
# Health endpoint
GET /health

# System status
GET /api/system/status

# Metrics
GET /metrics
```

### **Performance Metrics**
- Response time: <3s target
- Memory usage: Monitor for leaks
- CPU usage: Scale based on load
- Error rates: Alert on >5%

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Backend Won't Start**
```bash
# Check Python version (3.8+ required)
python --version

# Install missing dependencies
pip install -r requirements.txt

# Check port availability
netstat -an | findstr :5000
```

#### **Frontend Won't Start**
```bash
# Check Node.js version (14+ required)
node --version

# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Check port availability
netstat -an | findstr :3000
```

#### **API Connection Issues**
- Verify backend is running on port 5000
- Check CORS configuration
- Verify API URL in frontend environment

#### **Docker Issues**
```bash
# Check Docker status
docker ps

# View container logs
docker logs nasa-exoplanet-api

# Restart services
docker-compose restart
```

---

## 🎯 **Quick Start Summary**

### **Fastest Deployment (Local)**
```bash
# Terminal 1 - Backend
cd backend && pip install -r requirements.txt && python app.py

# Terminal 2 - Frontend  
cd frontend && npm install && npm start
```

### **Production Deployment (Docker)**
```bash
docker-compose up -d --build
```

### **Cloud Deployment (Render)**
```bash
git push origin main
# Then connect repository in Render dashboard
```

---

## 🎉 **Success Indicators**

### **✅ Deployment Successful When:**
- Backend responds at `/health` endpoint
- Frontend loads at `http://localhost:3000`
- File upload works correctly
- Exoplanet analysis returns results
- Results are consistent across requests

### **🌌 Ready for NASA Space Apps Challenge!**

Your NASA exoplanet detection API is now deployed and ready to discover new worlds! 🚀🌍
