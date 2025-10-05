# 🚀 Quick Deployment - NASA Exoplanet API

## **3 Simple Steps to Deploy**

### **Step 1: Start Backend** ⚡
```bash
cd backend
pip install -r requirements.txt
python app.py
```
✅ Backend running at: `http://localhost:5000`

### **Step 2: Start Frontend** 🌐
```bash
cd frontend
npm install
npm start
```
✅ Frontend running at: `http://localhost:3000`

### **Step 3: Test the System** 🧪
1. Open `http://localhost:3000`
2. Upload a CSV file with exoplanet data
3. Click "Start Analysis"
4. View results!

---

## **Alternative: Docker Deployment** 🐳

### **One Command Deployment**
```bash
docker-compose up --build
```

✅ **Both services will be running:**
- Backend: `http://localhost:5000`
- Frontend: `http://localhost:3000`

---

## **Cloud Deployment (Render.com)** ☁️

### **Steps:**
1. **Push to GitHub**: `git push origin main`
2. **Connect to Render**: Link your GitHub repo
3. **Auto-Deploy**: Render will use `render.yaml`

---

## **Verify Deployment** ✅

### **Health Check:**
```bash
curl http://localhost:5000/health
```

### **Test Upload:**
- Go to `http://localhost:3000`
- Upload any CSV with columns: `koi_fpflag_nt`, `koi_fpflag_co`, `koi_prad`
- Get instant exoplanet detection results!

---

## **🎉 You're Ready for NASA Space Apps Challenge!**

Your exoplanet detection API is now live and ready to discover new worlds! 🌌🚀
