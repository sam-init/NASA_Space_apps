import axios from 'axios';

// Base URL for the Flask backend
const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000, // 30 seconds timeout for file uploads
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth tokens if needed
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    console.error('API Error:', error);
    
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      
      switch (status) {
        case 401:
          // Unauthorized - redirect to login or clear token
          localStorage.removeItem('authToken');
          break;
        case 429:
          // Rate limited
          throw new Error('Too many requests. Please try again later.');
        case 500:
          throw new Error('Server error. Please try again later.');
        default:
          throw new Error(data.message || 'An error occurred');
      }
    } else if (error.request) {
      // Network error
      throw new Error('Network error. Please check your connection.');
    } else {
      throw new Error('An unexpected error occurred');
    }
  }
);

// API endpoints
export const apiEndpoints = {
  // File upload and analysis
  uploadFile: (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);
    
    return api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress) {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress(percentCompleted);
        }
      },
    });
  },

  // Analyze uploaded data
  analyzeData: (fileId, options = {}) => {
    return api.post('/analyze', {
      file_id: fileId,
      ...options,
    });
  },

  // Get analysis results
  getResults: (analysisId) => {
    return api.get(`/results/${analysisId}`);
  },

  // Get all user's analyses
  getAllAnalyses: (page = 1, limit = 10) => {
    return api.get('/analyses', {
      params: { page, limit },
    });
  },

  // Get dashboard statistics
  getDashboardStats: () => {
    return api.get('/dashboard/stats');
  },

  // Get sample datasets
  getSampleDatasets: () => {
    return api.get('/datasets/samples');
  },

  // Download sample dataset
  downloadSampleDataset: (datasetId) => {
    return api.get(`/datasets/download/${datasetId}`, {
      responseType: 'blob',
    });
  },

  // Get system status
  getSystemStatus: () => {
    return api.get('/system/status');
  },

  // Export results
  exportResults: (analysisId, format = 'json') => {
    return api.get(`/results/${analysisId}/export`, {
      params: { format },
      responseType: format === 'pdf' ? 'blob' : 'json',
    });
  },

  // Delete analysis
  deleteAnalysis: (analysisId) => {
    return api.delete(`/analyses/${analysisId}`);
  },

  // Get light curve data for visualization
  getLightCurveData: (analysisId) => {
    return api.get(`/results/${analysisId}/lightcurve`);
  },

  // Batch analysis
  batchAnalyze: (fileIds, options = {}) => {
    return api.post('/analyze/batch', {
      file_ids: fileIds,
      ...options,
    });
  },

  // Get model information
  getModelInfo: () => {
    return api.get('/model/info');
  },

  // Health check
  healthCheck: () => {
    return api.get('/health');
  },
};

// Utility functions
export const utils = {
  // Format file size
  formatFileSize: (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  // Validate file type
  validateFileType: (file) => {
    const allowedTypes = ['.csv', '.fits', '.dat', '.txt'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    return allowedTypes.includes(fileExtension);
  },

  // Validate file size (max 100MB)
  validateFileSize: (file, maxSizeMB = 100) => {
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    return file.size <= maxSizeBytes;
  },

  // Format confidence percentage
  formatConfidence: (confidence) => {
    return `${confidence.toFixed(1)}%`;
  },

  // Format timestamp
  formatTimestamp: (timestamp) => {
    return new Date(timestamp).toLocaleString();
  },

  // Generate unique ID for tracking uploads
  generateUploadId: () => {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  },
};

// Error classes
export class APIError extends Error {
  constructor(message, status, code) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.code = code;
  }
}

export class ValidationError extends Error {
  constructor(message, field) {
    super(message);
    this.name = 'ValidationError';
    this.field = field;
  }
}

// Export the configured axios instance as default
export default api;
