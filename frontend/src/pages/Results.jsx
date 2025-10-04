import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Star, 
  Download, 
  Share2, 
  Eye, 
  TrendingUp,
  Calendar,
  Clock,
  Target,
  BarChart3,
  Globe,
  Zap
} from 'lucide-react';

const Results = () => {
  const [selectedResult, setSelectedResult] = useState(0);

  const analysisResults = [
    {
      id: 1,
      filename: "kepler_452b_candidate.csv",
      timestamp: "2024-10-04 14:30:22",
      exoplanetDetected: true,
      confidence: 96.7,
      transitDepth: 0.0084,
      period: 384.8,
      duration: 10.4,
      starMagnitude: 13.426,
      planetRadius: 1.63,
      equilibriumTemp: 265,
      status: "confirmed"
    },
    {
      id: 2,
      filename: "tess_toi_715b.dat",
      timestamp: "2024-10-04 13:15:45",
      exoplanetDetected: true,
      confidence: 94.2,
      transitDepth: 0.0056,
      period: 17.66,
      duration: 3.2,
      starMagnitude: 12.18,
      planetRadius: 1.55,
      equilibriumTemp: 460,
      status: "candidate"
    },
    {
      id: 3,
      filename: "k2_false_positive.csv",
      timestamp: "2024-10-04 12:45:12",
      exoplanetDetected: false,
      confidence: 23.1,
      transitDepth: null,
      period: null,
      duration: null,
      starMagnitude: 14.2,
      planetRadius: null,
      equilibriumTemp: null,
      status: "false_positive"
    }
  ];

  const currentResult = analysisResults[selectedResult];

  return (
    <div className="min-h-screen bg-black pt-20 pb-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-8"
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="gradient-text">Analysis Results</span>
          </h1>
          <p className="text-xl text-gray-400">
            Detailed exoplanet detection results and visualizations
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Results List */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="lg:col-span-1"
          >
            <div className="glass p-6 rounded-2xl">
              <h2 className="text-xl font-bold text-white mb-4">Recent Results</h2>
              
              <div className="space-y-3">
                {analysisResults.map((result, index) => (
                  <motion.div
                    key={result.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 * index }}
                    onClick={() => setSelectedResult(index)}
                    className={`p-4 rounded-xl cursor-pointer transition-all duration-300 ${
                      selectedResult === index 
                        ? 'bg-space-500/20 border border-space-500/30' 
                        : 'bg-white/5 hover:bg-white/10'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className={`w-3 h-3 rounded-full ${
                        result.exoplanetDetected 
                          ? 'bg-green-500' 
                          : 'bg-red-500'
                      }`} />
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        result.status === 'confirmed' 
                          ? 'bg-green-500/20 text-green-400'
                          : result.status === 'candidate'
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {result.status.replace('_', ' ')}
                      </span>
                    </div>
                    
                    <h3 className="font-medium text-white text-sm mb-1 truncate">
                      {result.filename}
                    </h3>
                    
                    <div className="flex items-center space-x-2 text-xs text-gray-400">
                      <Clock className="h-3 w-3" />
                      <span>{new Date(result.timestamp).toLocaleDateString()}</span>
                    </div>
                    
                    <div className="mt-2">
                      <span className={`text-sm font-medium ${
                        result.exoplanetDetected ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {result.confidence.toFixed(1)}% confidence
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Main Result Display */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="lg:col-span-3 space-y-6"
          >
            {/* Result Header */}
            <div className="glass p-6 rounded-2xl">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-white mb-2">{currentResult.filename}</h2>
                  <div className="flex items-center space-x-4 text-gray-400">
                    <div className="flex items-center space-x-1">
                      <Calendar className="h-4 w-4" />
                      <span className="text-sm">{new Date(currentResult.timestamp).toLocaleString()}</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Target className="h-4 w-4" />
                      <span className="text-sm">{currentResult.confidence.toFixed(1)}% confidence</span>
                    </div>
                  </div>
                </div>
                
                <div className="flex space-x-3">
                  <button className="btn-secondary">
                    <Eye className="mr-2 h-4 w-4" />
                    View Data
                  </button>
                  <button className="btn-secondary">
                    <Share2 className="mr-2 h-4 w-4" />
                    Share
                  </button>
                  <button className="btn-primary">
                    <Download className="mr-2 h-4 w-4" />
                    Download
                  </button>
                </div>
              </div>

              {/* Detection Status */}
              <div className={`p-6 rounded-xl ${
                currentResult.exoplanetDetected 
                  ? 'bg-green-500/20 border border-green-500/30' 
                  : 'bg-red-500/20 border border-red-500/30'
              }`}>
                <div className="flex items-center space-x-3 mb-4">
                  <Star className={`h-8 w-8 ${
                    currentResult.exoplanetDetected ? 'text-green-400' : 'text-red-400'
                  }`} />
                  <div>
                    <h3 className={`text-2xl font-bold ${
                      currentResult.exoplanetDetected ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {currentResult.exoplanetDetected ? 'Exoplanet Detected!' : 'No Transit Detected'}
                    </h3>
                    <p className="text-gray-300">
                      Analysis completed with {currentResult.confidence.toFixed(1)}% confidence
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Detailed Parameters */}
            {currentResult.exoplanetDetected && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Transit Properties */}
                <div className="glass p-6 rounded-2xl">
                  <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                    <TrendingUp className="mr-2 h-5 w-5 text-space-400" />
                    Transit Properties
                  </h3>
                  
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Transit Depth</span>
                      <span className="text-white font-mono">
                        {(currentResult.transitDepth * 100).toFixed(4)}%
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Orbital Period</span>
                      <span className="text-white font-mono">
                        {currentResult.period.toFixed(2)} days
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Transit Duration</span>
                      <span className="text-white font-mono">
                        {currentResult.duration.toFixed(1)} hours
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Star Magnitude</span>
                      <span className="text-white font-mono">
                        {currentResult.starMagnitude.toFixed(3)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Planet Properties */}
                <div className="glass p-6 rounded-2xl">
                  <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                    <Globe className="mr-2 h-5 w-5 text-cosmic-400" />
                    Planet Properties
                  </h3>
                  
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Planet Radius</span>
                      <span className="text-white font-mono">
                        {currentResult.planetRadius.toFixed(2)} R⊕
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Equilibrium Temperature</span>
                      <span className="text-white font-mono">
                        {currentResult.equilibriumTemp} K
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Classification</span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        currentResult.planetRadius < 1.25 
                          ? 'bg-blue-500/20 text-blue-400'
                          : currentResult.planetRadius < 2.0
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-orange-500/20 text-orange-400'
                      }`}>
                        {currentResult.planetRadius < 1.25 
                          ? 'Earth-like' 
                          : currentResult.planetRadius < 2.0 
                          ? 'Super-Earth' 
                          : 'Mini-Neptune'
                        }
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Habitability Zone</span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        currentResult.equilibriumTemp > 200 && currentResult.equilibriumTemp < 350
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {currentResult.equilibriumTemp > 200 && currentResult.equilibriumTemp < 350
                          ? 'Potentially Habitable'
                          : 'Outside Habitable Zone'
                        }
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Light Curve Visualization Placeholder */}
            <div className="glass p-6 rounded-2xl">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                <BarChart3 className="mr-2 h-5 w-5 text-space-400" />
                Light Curve Analysis
              </h3>
              
              <div className="bg-gray-900/50 rounded-xl p-8 text-center">
                <div className="w-full h-64 bg-gradient-to-r from-space-500/20 to-cosmic-500/20 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <BarChart3 className="h-16 w-16 text-gray-600 mx-auto mb-4" />
                    <p className="text-gray-400 text-lg">Interactive Light Curve Visualization</p>
                    <p className="text-gray-500 text-sm mt-2">
                      Chart component will be integrated with backend data
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Analysis Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="glass p-6 rounded-2xl text-center">
                <Zap className="h-8 w-8 text-yellow-400 mx-auto mb-3" />
                <h4 className="text-lg font-bold text-white mb-2">Processing Time</h4>
                <p className="text-2xl font-bold gradient-text">2.3s</p>
                <p className="text-sm text-gray-400">Lightning fast analysis</p>
              </div>
              
              <div className="glass p-6 rounded-2xl text-center">
                <Target className="h-8 w-8 text-green-400 mx-auto mb-3" />
                <h4 className="text-lg font-bold text-white mb-2">Model Accuracy</h4>
                <p className="text-2xl font-bold gradient-text">99.2%</p>
                <p className="text-sm text-gray-400">On validation dataset</p>
              </div>
              
              <div className="glass p-6 rounded-2xl text-center">
                <Star className="h-8 w-8 text-purple-400 mx-auto mb-3" />
                <h4 className="text-lg font-bold text-white mb-2">Data Points</h4>
                <p className="text-2xl font-bold gradient-text">15,247</p>
                <p className="text-sm text-gray-400">Analyzed in this session</p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Results;
