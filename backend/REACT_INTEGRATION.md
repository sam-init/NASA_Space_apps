# 🚀 React Integration Guide

**Real-time Exoplanet Analysis API for React Frontend**

## 📡 API Endpoints

### Base URL: `http://localhost:5000`

## 🔧 React Integration Examples

### 1. File Upload Component

```jsx
import React, { useState } from 'react';
import axios from 'axios';

const ExoplanetUpload = () => {
  const [file, setFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileUpload = async (event) => {
    event.preventDefault();
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(
        'http://localhost:5000/api/upload',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      setUploadResult(response.data);
      console.log('Upload successful:', response.data);
    } catch (error) {
      console.error('Upload failed:', error.response?.data || error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <h2>Upload Exoplanet Data</h2>
      <form onSubmit={handleFileUpload}>
        <input
          type="file"
          accept=".csv,.json,.fits,.dat,.txt"
          onChange={(e) => setFile(e.target.files[0])}
          disabled={loading}
        />
        <button type="submit" disabled={!file || loading}>
          {loading ? 'Uploading...' : 'Upload File'}
        </button>
      </form>

      {uploadResult && (
        <div className="upload-result">
          <h3>Upload Result</h3>
          <p><strong>File ID:</strong> {uploadResult.file_id}</p>
          <p><strong>Format:</strong> {uploadResult.format}</p>
          <p><strong>Valid:</strong> {uploadResult.validation?.valid ? '✅' : '❌'}</p>
          {uploadResult.validation?.nasa_format && (
            <p><strong>NASA Format:</strong> 🚀 Detected</p>
          )}
        </div>
      )}
    </div>
  );
};

export default ExoplanetUpload;
```

### 2. Real-time Analysis Component

```jsx
import React, { useState } from 'react';
import axios from 'axios';

const ExoplanetAnalysis = ({ fileId }) => {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeData = async () => {
    if (!fileId) return;

    setLoading(true);
    setError(null);

    const analysisRequest = {
      file_id: fileId,
      analysis_options: {
        model_type: 'nasa_pipeline',
        confidence_threshold: 0.5,
        include_feature_importance: true
      }
    };

    try {
      const startTime = Date.now();
      const response = await axios.post(
        'http://localhost:5000/api/analyze',
        analysisRequest,
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );
      const processingTime = (Date.now() - startTime) / 1000;

      setAnalysisResult({
        ...response.data,
        clientProcessingTime: processingTime
      });
      
      console.log('Analysis completed:', response.data);
    } catch (error) {
      setError(error.response?.data?.message || error.message);
      console.error('Analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="analysis-container">
      <h2>Exoplanet Analysis</h2>
      
      <button 
        onClick={analyzeData} 
        disabled={!fileId || loading}
        className="analyze-button"
      >
        {loading ? 'Analyzing...' : 'Analyze for Exoplanets'}
      </button>

      {error && (
        <div className="error-message">
          <h3>❌ Analysis Failed</h3>
          <p>{error}</p>
        </div>
      )}

      {analysisResult && (
        <div className="analysis-result">
          <h3>🔬 Analysis Results</h3>
          
          {/* Main Result */}
          <div className={`detection-result ${analysisResult.exoplanet_detected ? 'detected' : 'not-detected'}`}>
            <h4>
              {analysisResult.exoplanet_detected ? '🌍 Exoplanet Detected!' : '❌ No Exoplanet Detected'}
            </h4>
            <p><strong>Confidence:</strong> {analysisResult.confidence.toFixed(1)}%</p>
            <p><strong>Processing Time:</strong> {analysisResult.processing_time?.toFixed(2)}s</p>
          </div>

          {/* Transit Parameters */}
          {analysisResult.exoplanet_detected && analysisResult.transit_parameters && (
            <div className="transit-parameters">
              <h4>🪐 Transit Parameters</h4>
              <div className="parameter-grid">
                <div className="parameter">
                  <strong>Planet Type:</strong> {analysisResult.transit_parameters.planet_type}
                </div>
                <div className="parameter">
                  <strong>Orbital Period:</strong> {analysisResult.transit_parameters.orbital_period_days} days
                </div>
                <div className="parameter">
                  <strong>Planet Radius:</strong> {analysisResult.transit_parameters.planet_radius_earth_radii} Earth radii
                </div>
                <div className="parameter">
                  <strong>Temperature:</strong> {analysisResult.transit_parameters.equilibrium_temperature_k} K
                </div>
                <div className="parameter">
                  <strong>Habitable Zone:</strong> {analysisResult.transit_parameters.habitable_zone ? '✅ Yes' : '❌ No'}
                </div>
              </div>
            </div>
          )}

          {/* Feature Importance */}
          {analysisResult.feature_importance && Object.keys(analysisResult.feature_importance).length > 0 && (
            <div className="feature-importance">
              <h4>📊 Feature Importance</h4>
              <div className="importance-bars">
                {Object.entries(analysisResult.feature_importance).map(([feature, importance]) => (
                  <div key={feature} className="importance-bar">
                    <span className="feature-name">{feature}</span>
                    <div className="bar-container">
                      <div 
                        className="bar-fill" 
                        style={{ width: `${importance * 100}%` }}
                      ></div>
                    </div>
                    <span className="importance-value">{(importance * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ExoplanetAnalysis;
```

### 3. Results Detail Component

```jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ExoplanetResults = ({ analysisId }) => {
  const [detailedResults, setDetailedResults] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (analysisId) {
      fetchDetailedResults();
    }
  }, [analysisId]);

  const fetchDetailedResults = async () => {
    setLoading(true);
    try {
      const response = await axios.get(
        `http://localhost:5000/api/results/${analysisId}`
      );
      setDetailedResults(response.data);
    } catch (error) {
      console.error('Failed to fetch detailed results:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div>Loading detailed results...</div>;
  if (!detailedResults) return null;

  return (
    <div className="detailed-results">
      <h3>📈 Detailed Analysis Results</h3>
      
      {/* Interpretation */}
      {detailedResults.interpretation && (
        <div className="interpretation">
          <h4>💡 Interpretation</h4>
          <p><strong>Summary:</strong> {detailedResults.interpretation.summary}</p>
          <p><strong>Reliability:</strong> {detailedResults.interpretation.reliability}</p>
          <p><strong>Recommendation:</strong> {detailedResults.interpretation.recommendation}</p>
        </div>
      )}

      {/* Model Performance */}
      {detailedResults.model_performance && (
        <div className="model-performance">
          <h4>🎯 Model Performance</h4>
          <p><strong>Accuracy:</strong> {(detailedResults.model_performance.accuracy * 100).toFixed(1)}%</p>
          <p><strong>Precision:</strong> {(detailedResults.model_performance.precision * 100).toFixed(1)}%</p>
          <p><strong>Recall:</strong> {(detailedResults.model_performance.recall * 100).toFixed(1)}%</p>
        </div>
      )}

      {/* Metadata */}
      <div className="metadata">
        <h4>📋 Analysis Metadata</h4>
        <p><strong>Analysis ID:</strong> {detailedResults.analysis_id}</p>
        <p><strong>Model Used:</strong> {detailedResults.model_used}</p>
        <p><strong>API Version:</strong> {detailedResults.api_version}</p>
        <p><strong>Timestamp:</strong> {new Date(detailedResults.timestamp).toLocaleString()}</p>
      </div>
    </div>
  );
};

export default ExoplanetResults;
```

### 4. Complete Integration Example

```jsx
import React, { useState } from 'react';
import ExoplanetUpload from './components/ExoplanetUpload';
import ExoplanetAnalysis from './components/ExoplanetAnalysis';
import ExoplanetResults from './components/ExoplanetResults';
import './App.css';

function App() {
  const [currentFileId, setCurrentFileId] = useState(null);
  const [currentAnalysisId, setCurrentAnalysisId] = useState(null);

  const handleUploadComplete = (uploadResult) => {
    if (uploadResult.ready_for_analysis) {
      setCurrentFileId(uploadResult.file_id);
    }
  };

  const handleAnalysisComplete = (analysisResult) => {
    setCurrentAnalysisId(analysisResult.analysis_id);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🌌 ExoPlanet AI</h1>
        <p>NASA Space Apps Challenge 2025 - Team BrainRot</p>
      </header>

      <main className="App-main">
        <div className="workflow-container">
          {/* Step 1: Upload */}
          <div className="workflow-step">
            <ExoplanetUpload onUploadComplete={handleUploadComplete} />
          </div>

          {/* Step 2: Analysis */}
          {currentFileId && (
            <div className="workflow-step">
              <ExoplanetAnalysis 
                fileId={currentFileId} 
                onAnalysisComplete={handleAnalysisComplete}
              />
            </div>
          )}

          {/* Step 3: Detailed Results */}
          {currentAnalysisId && (
            <div className="workflow-step">
              <ExoplanetResults analysisId={currentAnalysisId} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
```

### 5. CSS Styles

```css
/* App.css */
.App {
  text-align: center;
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.App-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 30px;
  border-radius: 10px;
  margin-bottom: 30px;
}

.workflow-container {
  display: flex;
  flex-direction: column;
  gap: 30px;
}

.workflow-step {
  background: #f8f9fa;
  padding: 25px;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.upload-container, .analysis-container {
  text-align: left;
}

.detection-result {
  padding: 20px;
  border-radius: 8px;
  margin: 15px 0;
}

.detection-result.detected {
  background: #d4edda;
  border: 2px solid #28a745;
}

.detection-result.not-detected {
  background: #f8d7da;
  border: 2px solid #dc3545;
}

.parameter-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 15px;
  margin-top: 15px;
}

.parameter {
  background: white;
  padding: 10px;
  border-radius: 5px;
  border-left: 4px solid #667eea;
}

.importance-bars {
  margin-top: 15px;
}

.importance-bar {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
  gap: 10px;
}

.feature-name {
  min-width: 120px;
  font-weight: bold;
}

.bar-container {
  flex: 1;
  height: 20px;
  background: #e9ecef;
  border-radius: 10px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #28a745, #20c997);
  transition: width 0.3s ease;
}

.importance-value {
  min-width: 50px;
  text-align: right;
  font-weight: bold;
}

.error-message {
  background: #f8d7da;
  color: #721c24;
  padding: 15px;
  border-radius: 5px;
  border: 1px solid #f5c6cb;
}

.analyze-button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 6px;
  font-size: 16px;
  cursor: pointer;
  transition: transform 0.2s;
}

.analyze-button:hover:not(:disabled) {
  transform: translateY(-2px);
}

.analyze-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
```

## 🔧 Setup Instructions

### 1. Install Dependencies

```bash
npm install axios
```

### 2. Configure CORS

The backend is already configured with CORS for:
- `http://localhost:3000` (default React dev server)
- `http://localhost:3001` (alternative port)

### 3. Environment Variables

Create `.env` file in your React project:

```env
REACT_APP_API_BASE_URL=http://localhost:5000
```

### 4. API Service Helper

```javascript
// services/api.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000, // 10 second timeout
});

export const exoplanetAPI = {
  uploadFile: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/api/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },

  analyzeData: (fileId, options = {}) => {
    return api.post('/api/analyze', {
      file_id: fileId,
      analysis_options: {
        model_type: 'nasa_pipeline',
        confidence_threshold: 0.5,
        include_feature_importance: true,
        ...options
      }
    });
  },

  getResults: (analysisId) => {
    return api.get(`/api/results/${analysisId}`);
  },

  getSystemStatus: () => {
    return api.get('/api/system/status');
  }
};

export default api;
```

## 🚀 Quick Start

1. **Start the Flask backend:**
   ```bash
   cd backend
   python app.py
   ```

2. **Start your React app:**
   ```bash
   npm start
   ```

3. **Test the integration:**
   - Upload a CSV file with exoplanet data
   - Click "Analyze for Exoplanets"
   - View real-time results with transit parameters
   - Check detailed analysis results

## 📊 Expected Response Format

```json
{
  "exoplanet_detected": true,
  "confidence": 87.3,
  "transit_parameters": {
    "orbital_period_days": 234.5,
    "transit_depth_ppm": 1250.0,
    "planet_radius_earth_radii": 1.2,
    "equilibrium_temperature_k": 315.0,
    "habitable_zone": true,
    "planet_type": "Earth-like"
  },
  "feature_importance": {
    "koi_fpflag_nt": 0.35,
    "koi_fpflag_co": 0.20,
    "koi_prad": 0.25
  },
  "processing_time": 1.23,
  "analysis_id": "uuid-string"
}
```

🎉 **Your React frontend is now ready for real-time exoplanet detection!**
