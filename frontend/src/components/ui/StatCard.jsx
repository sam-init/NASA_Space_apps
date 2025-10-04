import React from 'react';
import { motion } from 'framer-motion';

const StatCard = ({ number, label, icon: Icon }) => {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      transition={{ duration: 0.3 }}
      className="glass p-6 rounded-2xl text-center group hover:bg-white/10 transition-all duration-300"
    >
      <div className="space-y-4">
        <div className="flex justify-center">
          <div className="relative">
            <Icon className="h-8 w-8 text-space-400 group-hover:text-space-300 transition-colors duration-300" />
            <div className="absolute inset-0 bg-space-400 opacity-20 blur-lg group-hover:opacity-30 transition-opacity duration-300"></div>
          </div>
        </div>
        
        <div className="space-y-2">
          <motion.div
            initial={{ opacity: 0, scale: 0.5 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            viewport={{ once: true }}
            className="text-3xl md:text-4xl font-bold gradient-text"
          >
            {number}
          </motion.div>
          
          <p className="text-gray-400 font-medium text-sm uppercase tracking-wider">
            {label}
          </p>
        </div>
      </div>
    </motion.div>
  );
};

export default StatCard;
