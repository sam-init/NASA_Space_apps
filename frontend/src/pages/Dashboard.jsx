import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart3, 
  TrendingUp, 
  Database, 
  Clock, 
  Star,
  Globe,
  Activity,
  Zap
} from 'lucide-react';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalAnalyses: 1247,
    exoplanetsFound: 89,
    accuracyRate: 99.2,
    processingTime: 2.3
  });

  const recentAnalyses = [
    {
      id: 1,
      filename: "kepler_lightcurve_001.csv",
      timestamp: "2 minutes ago",
      result: "Exoplanet Detected",
      confidence: 94.7,
      status: "completed"
    },
    {
      id: 2,
      filename: "tess_sector_42.dat",
      timestamp: "15 minutes ago", 
      result: "No Transit Detected",
      confidence: 87.2,
      status: "completed"
    },
    {
      id: 3,
      filename: "k2_campaign_19.csv",
      timestamp: "1 hour ago",
      result: "Exoplanet Detected",
      confidence: 96.1,
      status: "completed"
    },
    {
      id: 4,
      filename: "custom_data_upload.csv",
      timestamp: "2 hours ago",
      result: "Processing...",
      confidence: null,
      status: "processing"
    }
  ];

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
            <span className="gradient-text">Dashboard</span>
          </h1>
          <p className="text-xl text-gray-400">
            Monitor your exoplanet detection analyses and system performance
          </p>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
            className="glass p-6 rounded-2xl"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center">
                <BarChart3 className="h-6 w-6 text-white" />
              </div>
              <span className="text-green-400 text-sm font-medium">+12%</span>
            </div>
            <h3 className="text-2xl font-bold text-white mb-1">{stats.totalAnalyses.toLocaleString()}</h3>
            <p className="text-gray-400 text-sm">Total Analyses</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="glass p-6 rounded-2xl"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center">
                <Star className="h-6 w-6 text-white" />
              </div>
              <span className="text-green-400 text-sm font-medium">+8</span>
            </div>
            <h3 className="text-2xl font-bold text-white mb-1">{stats.exoplanetsFound}</h3>
            <p className="text-gray-400 text-sm">Exoplanets Found</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="glass p-6 rounded-2xl"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-green-600 rounded-xl flex items-center justify-center">
                <TrendingUp className="h-6 w-6 text-white" />
              </div>
              <span className="text-green-400 text-sm font-medium">+0.3%</span>
            </div>
            <h3 className="text-2xl font-bold text-white mb-1">{stats.accuracyRate}%</h3>
            <p className="text-gray-400 text-sm">Accuracy Rate</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="glass p-6 rounded-2xl"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-orange-600 rounded-xl flex items-center justify-center">
                <Zap className="h-6 w-6 text-white" />
              </div>
              <span className="text-green-400 text-sm font-medium">-0.1s</span>
            </div>
            <h3 className="text-2xl font-bold text-white mb-1">{stats.processingTime}s</h3>
            <p className="text-gray-400 text-sm">Avg Processing Time</p>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Recent Analyses */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="lg:col-span-2"
          >
            <div className="glass p-6 rounded-2xl">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">Recent Analyses</h2>
                <Activity className="h-6 w-6 text-space-400" />
              </div>
              
              <div className="space-y-4">
                {recentAnalyses.map((analysis, index) => (
                  <motion.div
                    key={analysis.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 * index }}
                    className="flex items-center justify-between p-4 bg-white/5 rounded-xl hover:bg-white/10 transition-colors duration-300"
                  >
                    <div className="flex-1">
                      <h3 className="font-semibold text-white mb-1">{analysis.filename}</h3>
                      <p className="text-sm text-gray-400">{analysis.timestamp}</p>
                    </div>
                    
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <p className={`font-medium ${
                          analysis.result === "Exoplanet Detected" 
                            ? "text-green-400" 
                            : analysis.result === "Processing..." 
                            ? "text-yellow-400" 
                            : "text-gray-400"
                        }`}>
                          {analysis.result}
                        </p>
                        {analysis.confidence && (
                          <p className="text-sm text-gray-500">{analysis.confidence}% confidence</p>
                        )}
                      </div>
                      
                      <div className={`w-3 h-3 rounded-full ${
                        analysis.status === "completed" 
                          ? "bg-green-500" 
                          : "bg-yellow-500 animate-pulse"
                      }`} />
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Quick Actions & System Status */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="space-y-6"
          >
            {/* Quick Actions */}
            <div className="glass p-6 rounded-2xl">
              <h2 className="text-xl font-bold text-white mb-4">Quick Actions</h2>
              <div className="space-y-3">
                <button className="w-full btn-primary text-left">
                  <Database className="mr-3 h-5 w-5" />
                  Upload New Data
                </button>
                <button className="w-full btn-secondary text-left">
                  <Globe className="mr-3 h-5 w-5" />
                  Browse NASA Datasets
                </button>
                <button className="w-full bg-white/5 hover:bg-white/10 text-white font-medium py-3 px-4 rounded-lg transition-colors duration-300 flex items-center">
                  <Clock className="mr-3 h-5 w-5" />
                  View Analysis History
                </button>
              </div>
            </div>

            {/* System Status */}
            <div className="glass p-6 rounded-2xl">
              <h2 className="text-xl font-bold text-white mb-4">System Status</h2>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">AI Model</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-green-400 text-sm">Online</span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Database</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-green-400 text-sm">Connected</span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">API Status</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-green-400 text-sm">Operational</span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Queue</span>
                  <span className="text-white text-sm">3 pending</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
