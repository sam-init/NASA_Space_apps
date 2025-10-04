import React from 'react';
import { motion } from 'framer-motion';

const FeatureCard = ({ icon: Icon, title, description }) => {
  return (
    <motion.div
      whileHover={{ y: -5, scale: 1.02 }}
      transition={{ duration: 0.3 }}
      className="glass p-6 rounded-2xl hover:bg-white/10 transition-all duration-300 group h-full"
    >
      <div className="flex flex-col items-center text-center space-y-4">
        <div className="relative">
          <div className="w-16 h-16 bg-gradient-to-br from-space-500 to-cosmic-500 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
            <Icon className="h-8 w-8 text-white" />
          </div>
          <div className="absolute inset-0 bg-gradient-to-br from-space-500 to-cosmic-500 rounded-2xl opacity-20 blur-lg group-hover:opacity-40 transition-opacity duration-300"></div>
        </div>
        
        <h3 className="text-xl font-bold text-white group-hover:text-space-300 transition-colors duration-300">
          {title}
        </h3>
        
        <p className="text-gray-400 leading-relaxed text-sm">
          {description}
        </p>
      </div>
    </motion.div>
  );
};

export default FeatureCard;
