import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Target, 
  Brain, 
  Database, 
  Zap, 
  Globe, 
  Star, 
  Rocket,
  ChevronRight,
  Play,
  CheckCircle,
  Users,
  Award,
  ArrowRight
} from 'lucide-react';
import GalaxyBackground from '../components/ui/GalaxyBackground';
import FeatureCard from '../components/ui/FeatureCard';
import StatCard from '../components/ui/StatCard';

const LandingPage = () => {
  const features = [
    {
      icon: Brain,
      title: "AI-Powered Detection",
      description: "Advanced machine learning algorithms trained on NASA's Kepler, TESS, and K2 datasets for accurate exoplanet identification."
    },
    {
      icon: Database,
      title: "Real-time Analysis",
      description: "Upload your light curve data and get instant predictions with detailed confidence scores and visualizations."
    },
    {
      icon: Zap,
      title: "Lightning Fast",
      description: "Optimized neural networks provide results in seconds, not hours. Perfect for time-sensitive research."
    },
    {
      icon: Globe,
      title: "Open Source Data",
      description: "Built entirely on NASA's publicly available datasets, ensuring transparency and reproducibility."
    }
  ];

  const stats = [
    { number: "5,000+", label: "Exoplanets Detected", icon: Star },
    { number: "99.2%", label: "Accuracy Rate", icon: CheckCircle },
    { number: "500+", label: "Researchers Using", icon: Users },
    { number: "NASA", label: "Space Apps Winner", icon: Award }
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section with Galaxy Background */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <GalaxyBackground />
        
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
            className="space-y-8"
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2, duration: 0.8 }}
              className="inline-flex items-center space-x-2 glass px-4 py-2 rounded-full"
            >
              <Rocket className="h-4 w-4 text-space-400" />
              <span className="text-sm font-medium text-gray-300">NASA Space Apps Challenge 2025</span>
            </motion.div>

            {/* Main Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.8 }}
              className="text-5xl md:text-7xl font-bold leading-tight"
            >
              <span className="glow-text">Discover</span>
              <br />
              <span className="gradient-text">Exoplanets</span>
              <br />
              <span className="text-white">with AI</span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.8 }}
              className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto leading-relaxed"
            >
              Harness the power of artificial intelligence to analyze NASA's exoplanet data and uncover new worlds beyond our solar system.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8, duration: 0.8 }}
              className="flex flex-col sm:flex-row gap-4 justify-center items-center"
            >
              <Link to="/upload" className="btn-primary group">
                <span>Start Detection</span>
                <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform duration-300" />
              </Link>
              
              <button className="btn-secondary group">
                <Play className="mr-2 h-5 w-5" />
                <span>Watch Demo</span>
              </button>
            </motion.div>

            {/* Floating Animation */}
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              className="mt-12"
            >
              <Target className="h-16 w-16 text-space-400 mx-auto opacity-60" />
            </motion.div>
          </motion.div>
        </div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 1 }}
          className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-6 h-10 border-2 border-space-400 rounded-full flex justify-center"
          >
            <motion.div
              animate={{ y: [0, 12, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-1 h-3 bg-space-400 rounded-full mt-2"
            />
          </motion.div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gradient-to-b from-black to-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              <span className="gradient-text">Cutting-Edge Technology</span>
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Our AI system combines the latest in machine learning with NASA's comprehensive exoplanet datasets to deliver unprecedented accuracy in planetary detection.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <FeatureCard {...feature} />
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 bg-gradient-to-b from-gray-900 to-black">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              <span className="text-white">Proven</span> <span className="gradient-text">Results</span>
            </h2>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Join the growing community of researchers and astronomers using our platform to make groundbreaking discoveries.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <StatCard {...stat} />
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 bg-gradient-to-b from-black to-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              <span className="gradient-text">How It Works</span>
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Our streamlined process makes exoplanet detection accessible to researchers at all levels.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                step: "01",
                title: "Upload Data",
                description: "Upload your light curve data from telescopes or use our sample datasets from NASA's archives."
              },
              {
                step: "02", 
                title: "AI Analysis",
                description: "Our trained neural network analyzes the data for transit patterns indicating exoplanet presence."
              },
              {
                step: "03",
                title: "Get Results",
                description: "Receive detailed predictions with confidence scores, visualizations, and exportable reports."
              }
            ].map((item, index) => (
              <motion.div
                key={item.step}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="glass p-8 rounded-2xl hover:bg-white/10 transition-all duration-300 group">
                  <div className="text-6xl font-bold gradient-text mb-4 group-hover:scale-110 transition-transform duration-300">
                    {item.step}
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-4">{item.title}</h3>
                  <p className="text-gray-400 leading-relaxed">{item.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-b from-gray-900 to-black">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="space-y-8"
          >
            <h2 className="text-4xl md:text-5xl font-bold">
              <span className="text-white">Ready to</span> <span className="gradient-text">Explore?</span>
            </h2>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Join the next generation of space exploration. Start detecting exoplanets with AI today.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/upload" className="btn-primary group">
                <span>Start Your Discovery</span>
                <ChevronRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform duration-300" />
              </Link>
              <Link to="/dashboard" className="btn-secondary">
                View Dashboard
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
