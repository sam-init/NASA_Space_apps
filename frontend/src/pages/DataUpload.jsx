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
    
    try {
      // First upload the file
      const file = files[0].file;
      const formData = new FormData();
      formData.append('file', file);
      
      const uploadResponse = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (!uploadResponse.ok) {
        throw new Error('Upload failed');
      }
      
      const uploadData = await uploadResponse.json();
      const fileId = uploadData.file_id;
      
      // Then analyze the uploaded file
      const analysisResponse = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_id: fileId,
          analysis_options: {
            model_type: 'nasa_pipeline',
            confidence_threshold: 0.5,
            include_feature_importance: true
          }
        }),
      });
      
      if (!analysisResponse.ok) {
        throw new Error('Analysis failed');
      }
      
      const analysisData = await analysisResponse.json();
      
      // Convert backend response to frontend format
      setAnalysisResults({
        exoplanetDetected: analysisData.exoplanet_detected,
        confidence: analysisData.confidence,
        // Store as fraction (0..1). ppm -> fraction = ppm / 1,000,000
        transitDepth: ((analysisData.transit_parameters?.transit_depth_ppm || 0) / 1000000),
        period: analysisData.transit_parameters?.orbital_period_days || 0,
        duration: analysisData.transit_parameters?.transit_duration_hours || 0,
        planetType: analysisData.transit_parameters?.planet_type || 'Unknown',
        habitableZone: analysisData.transit_parameters?.habitable_zone || false,
        processingTime: analysisData.processing_time || 0,
        featureImportance: analysisData.feature_importance || {},
        analysisId: analysisData.analysis_id,
        rawData: analysisData
      });
      
    } catch (error) {
      console.error('Analysis error:', error);
      // Fallback to mock data if API fails
      setAnalysisResults({
        exoplanetDetected: false,
        confidence: 0,
        transitDepth: 0,
        period: 0,
        duration: 0,
        error: error.message
      });
    } finally {
      setUploading(false);
    }
  };

  const downloadReport = () => {
    if (!analysisResults) return;
    
    const reportData = {
      timestamp: new Date().toISOString(),
      analysis: {
        exoplanet_detected: analysisResults.exoplanetDetected,
        confidence: analysisResults.confidence,
        planet_type: analysisResults.planetType,
        habitable_zone: analysisResults.habitableZone
      },
      transit_parameters: {
        orbital_period_days: analysisResults.period,
        transit_depth_percent: (analysisResults.transitDepth * 100).toFixed(3),
        transit_duration_hours: analysisResults.duration
      },
      feature_importance: analysisResults.featureImportance,
      processing_time: analysisResults.processingTime,
      model_used: "NASA Exoplanet Pipeline"
    };
    
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `exoplanet_analysis_report_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadSampleDataset = (dataset) => {
    // Create sample data based on the dataset type
    let csvContent = '';
    
    if (dataset.name.includes('Kepler')) {
      csvContent = `koi_fpflag_nt,koi_fpflag_co,koi_fpflag_ss,koi_fpflag_ec,koi_prad,koi_disposition
0,0,0,0,1.2,CONFIRMED
1,0,0,0,2.5,FALSE POSITIVE
0,1,0,0,0.8,CONFIRMED
0,0,1,0,3.1,CANDIDATE
0,0,0,1,1.0,FALSE POSITIVE
0,0,0,0,0.9,CONFIRMED
1,1,0,0,4.2,FALSE POSITIVE
0,0,0,0,1.5,CONFIRMED`;
    } else if (dataset.name.includes('TESS')) {
      csvContent = `time,flux,flux_err
0.0,1.0000,0.0001
0.1,0.9999,0.0001
0.2,0.9995,0.0001
0.3,0.9998,0.0001
0.4,0.9997,0.0001
0.5,0.9999,0.0001
0.6,1.0001,0.0001
0.7,1.0000,0.0001`;
    } else {
      csvContent = `koi_fpflag_nt,koi_fpflag_co,koi_fpflag_ss,koi_fpflag_ec,koi_prad
0,0,0,0,1.1
0,0,0,0,0.83
1,0,0,0,2.4
0,1,0,0,1.6
0,0,0,0,1.04`;
    }
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${dataset.name.toLowerCase().replace(/\s+/g, '_')}_sample.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
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
                    <button 
                      onClick={downloadReport}
                      className="btn-primary"
                    >
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
                    onClick={() => downloadSampleDataset(dataset)}
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
