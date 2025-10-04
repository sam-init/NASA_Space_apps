import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  Upload, 
  FileText, 
  Database, 
  CheckCircle, 
  AlertCircle,
  X,
  Download,
  Play,
  Loader
} from 'lucide-react';

const DataUpload = () => {
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);

  // Sample NASA datasets
  const sampleDatasets = [
    {
      name: "Kepler Light Curves - Q1",
      description: "First quarter Kepler mission light curve data",
      size: "2.3 MB",
      type: "CSV"
    },
    {
      name: "TESS Sector 1-13",
      description: "TESS mission exoplanet candidates",
      size: "5.7 MB", 
      type: "FITS"
    },
    {
      name: "K2 Campaign 9",
      description: "K2 mission microlensing campaign data",
      size: "1.8 MB",
      type: "CSV"
    }
  ];

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  }, []);

  const handleFiles = (fileList) => {
    const newFiles = Array.from(fileList).map(file => ({
      file,
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      status: 'ready'
    }));
    setFiles(prev => [...prev, ...newFiles]);
  };

  const removeFile = (id) => {
    setFiles(prev => prev.filter(file => file.id !== id));
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleAnalysis = async () => {
    if (files.length === 0) return;
    
    setUploading(true);
    
    // Simulate analysis process
    setTimeout(() => {
      setAnalysisResults({
        exoplanetDetected: Math.random() > 0.5,
        confidence: Math.random() * 30 + 70,
        transitDepth: Math.random() * 0.02 + 0.001,
        period: Math.random() * 50 + 1,
        duration: Math.random() * 10 + 2
      });
      setUploading(false);
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-black pt-20 pb-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-8"
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">Data Upload</span>
          </h1>
          <p className="text-xl text-gray-400">
            Upload your light curve data or use NASA sample datasets for exoplanet detection
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="lg:col-span-2 space-y-6"
          >
            {/* File Upload Area */}
            <div className="glass p-6 rounded-2xl">
              <h2 className="text-2xl font-bold text-white mb-4">Upload Light Curve Data</h2>
              
              <div
                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
                  dragActive 
                    ? 'border-space-400 bg-space-500/10' 
                    : 'border-gray-600 hover:border-space-400'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  multiple
                  accept=".csv,.fits,.dat,.txt"
                  onChange={(e) => handleFiles(e.target.files)}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                
                <Upload className="h-12 w-12 text-space-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">
                  Drop your files here or click to browse
                </h3>
                <p className="text-gray-400 mb-4">
                  Supports CSV, FITS, DAT, and TXT formats
                </p>
                <p className="text-sm text-gray-500">
                  Maximum file size: 100MB per file
                </p>
              </div>
            </div>

            {/* Uploaded Files */}
            {files.length > 0 && (
              <div className="glass p-6 rounded-2xl">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-white">Uploaded Files</h3>
                  <span className="text-sm text-gray-400">{files.length} file(s)</span>
                </div>
                
                <div className="space-y-3">
                  {files.map((file) => (
                    <motion.div
                      key={file.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="flex items-center justify-between p-4 bg-white/5 rounded-xl"
                    >
                      <div className="flex items-center space-x-3">
                        <FileText className="h-8 w-8 text-space-400" />
                        <div>
                          <p className="font-medium text-white">{file.name}</p>
                          <p className="text-sm text-gray-400">{formatFileSize(file.size)}</p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <CheckCircle className="h-5 w-5 text-green-400" />
                        <button
                          onClick={() => removeFile(file.id)}
                          className="p-1 hover:bg-red-500/20 rounded-lg transition-colors duration-300"
                        >
                          <X className="h-4 w-4 text-red-400" />
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
                
                <div className="mt-6 flex space-x-4">
                  <button
                    onClick={handleAnalysis}
                    disabled={uploading}
                    className="btn-primary flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {uploading ? (
                      <>
                        <Loader className="mr-2 h-5 w-5 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-5 w-5" />
                        Start Analysis
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={() => setFiles([])}
                    className="btn-secondary"
                  >
                    Clear All
                  </button>
                </div>
              </div>
            )}

            {/* Analysis Results */}
            {analysisResults && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass p-6 rounded-2xl"
              >
                <h3 className="text-xl font-bold text-white mb-4">Analysis Results</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className={`p-4 rounded-xl ${
                      analysisResults.exoplanetDetected 
                        ? 'bg-green-500/20 border border-green-500/30' 
                        : 'bg-red-500/20 border border-red-500/30'
                    }`}>
                      <div className="flex items-center space-x-2 mb-2">
                        {analysisResults.exoplanetDetected ? (
                          <CheckCircle className="h-5 w-5 text-green-400" />
                        ) : (
                          <AlertCircle className="h-5 w-5 text-red-400" />
                        )}
                        <span className={`font-semibold ${
                          analysisResults.exoplanetDetected ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {analysisResults.exoplanetDetected ? 'Exoplanet Detected!' : 'No Transit Detected'}
                        </span>
                      </div>
                      <p className="text-sm text-gray-300">
                        Confidence: {analysisResults.confidence.toFixed(1)}%
                      </p>
                    </div>
                    
                    {analysisResults.exoplanetDetected && (
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Transit Depth:</span>
                          <span className="text-white">{(analysisResults.transitDepth * 100).toFixed(3)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Orbital Period:</span>
                          <span className="text-white">{analysisResults.period.toFixed(2)} days</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Transit Duration:</span>
                          <span className="text-white">{analysisResults.duration.toFixed(1)} hours</span>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex flex-col space-y-3">
                    <button className="btn-primary">
                      <Download className="mr-2 h-4 w-4" />
                      Download Report
                    </button>
                    <button className="btn-secondary">
                      View Detailed Analysis
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>

          {/* Sample Datasets Sidebar */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="space-y-6"
          >
            <div className="glass p-6 rounded-2xl">
              <div className="flex items-center space-x-2 mb-4">
                <Database className="h-6 w-6 text-space-400" />
                <h2 className="text-xl font-bold text-white">Sample Datasets</h2>
              </div>
              
              <p className="text-gray-400 text-sm mb-6">
                Try our AI with these curated NASA datasets
              </p>
              
              <div className="space-y-4">
                {sampleDatasets.map((dataset, index) => (
                  <motion.div
                    key={dataset.name}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 * index }}
                    className="p-4 bg-white/5 rounded-xl hover:bg-white/10 transition-colors duration-300 cursor-pointer group"
                  >
                    <h4 className="font-semibold text-white mb-2 group-hover:text-space-300 transition-colors duration-300">
                      {dataset.name}
                    </h4>
                    <p className="text-sm text-gray-400 mb-3">{dataset.description}</p>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-500">{dataset.size} • {dataset.type}</span>
                      <Download className="h-4 w-4 text-space-400 group-hover:text-space-300 transition-colors duration-300" />
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Upload Guidelines */}
            <div className="glass p-6 rounded-2xl">
              <h3 className="text-lg font-bold text-white mb-4">Upload Guidelines</h3>
              <div className="space-y-3 text-sm text-gray-400">
                <div className="flex items-start space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <span>CSV files with time and flux columns</span>
                </div>
                <div className="flex items-start space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <span>FITS files from Kepler/TESS missions</span>
                </div>
                <div className="flex items-start space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <span>Minimum 1000 data points recommended</span>
                </div>
                <div className="flex items-start space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <span>Time series data with regular cadence</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default DataUpload;
