#!/usr/bin/env python3
"""
Performance Optimizer for NASA Exoplanet API
NASA Space Apps Challenge 2025 - Team BrainRot

Optimizations for <3s inference time and production performance.
"""

import time
import functools
import logging
import threading
from typing import Dict, Any, Callable
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cachetools

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Performance optimization utilities for the NASA exoplanet API."""
    
    def __init__(self):
        self.cache = cachetools.TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.performance_metrics = {}
        
    def timing_decorator(self, func_name: str = None):
        """Decorator to measure function execution time."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                execution_time = end_time - start_time
                name = func_name or func.__name__
                
                # Store performance metrics
                if name not in self.performance_metrics:
                    self.performance_metrics[name] = []
                self.performance_metrics[name].append(execution_time)
                
                # Log slow operations
                if execution_time > 1.0:
                    logger.warning(f"Slow operation: {name} took {execution_time:.2f}s")
                
                return result
            return wrapper
        return decorator
    
    def cache_result(self, cache_key_func: Callable = None):
        """Decorator to cache function results."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    cache_key = str(hash((str(args), str(sorted(kwargs.items())))))
                
                # Check cache
                if cache_key in self.cache:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return self.cache[cache_key]
                
                # Execute and cache
                result = func(*args, **kwargs)
                self.cache[cache_key] = result
                logger.debug(f"Cached result for {func.__name__}")
                
                return result
            return wrapper
        return decorator
    
    def async_processing(self, func: Callable, *args, **kwargs):
        """Submit function for asynchronous processing."""
        future = self.executor.submit(func, *args, **kwargs)
        return future
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        stats = {}
        for func_name, times in self.performance_metrics.items():
            stats[func_name] = {
                'count': len(times),
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'total_time': np.sum(times)
            }
        return stats
    
    def clear_cache(self):
        """Clear the performance cache."""
        self.cache.clear()
        logger.info("Performance cache cleared")


# Global optimizer instance
optimizer = PerformanceOptimizer()


def optimize_dataframe_operations(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame operations for better performance."""
    # Use categorical data types for string columns with few unique values
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    # Convert to optimal numeric types
    for col in df.select_dtypes(include=['int64']):
        if df[col].min() >= 0 and df[col].max() <= 255:
            df[col] = df[col].astype('uint8')
        elif df[col].min() >= -128 and df[col].max() <= 127:
            df[col] = df[col].astype('int8')
        elif df[col].min() >= -32768 and df[col].max() <= 32767:
            df[col] = df[col].astype('int16')
        elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
            df[col] = df[col].astype('int32')
    
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df


@optimizer.timing_decorator('feature_extraction')
def optimized_feature_extraction(file_path: str) -> pd.DataFrame:
    """Optimized feature extraction with caching and performance monitoring."""
    try:
        # Read file with optimized settings
        if file_path.endswith('.csv'):
            # Use efficient CSV reading
            df = pd.read_csv(file_path, 
                           low_memory=False,
                           dtype_backend='numpy_nullable')
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Optimize DataFrame
        df = optimize_dataframe_operations(df)
        
        # Extract NASA KOI features efficiently
        nasa_features = ['koi_fpflag_nt', 'koi_fpflag_co', 'koi_fpflag_ss', 'koi_fpflag_ec', 'koi_prad']
        available_features = [col for col in nasa_features if col in df.columns]
        
        if available_features:
            # Use vectorized operations
            features_df = df[available_features].iloc[[0]].copy()
            
            # Fill missing values efficiently
            for col in nasa_features:
                if col not in features_df.columns:
                    if 'fpflag' in col:
                        features_df[col] = 0
                    else:
                        features_df[col] = 1.0
            
            return features_df[nasa_features]
        
        return None
        
    except Exception as e:
        logger.error(f"Optimized feature extraction failed: {e}")
        return None


@optimizer.timing_decorator('prediction')
@optimizer.cache_result(lambda features_df: str(features_df.values.tobytes()))
def optimized_prediction(features_df: pd.DataFrame) -> Dict[str, Any]:
    """Optimized prediction with caching."""
    from api.endpoints import generate_deterministic_prediction
    return generate_deterministic_prediction(features_df)


class FastInferenceEngine:
    """Fast inference engine for production deployment."""
    
    def __init__(self):
        self.model_cache = {}
        self.feature_cache = cachetools.LRUCache(maxsize=100)
        
    @optimizer.timing_decorator('fast_inference')
    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Fast prediction with multiple optimizations."""
        # Convert to numpy for faster operations
        features_array = features_df.values.astype(np.float32)
        
        # Create cache key
        cache_key = hash(features_array.tobytes())
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Fast deterministic calculation
        koi_fpflag_nt = features_array[0, 0]
        koi_fpflag_co = features_array[0, 1]
        koi_fpflag_ss = features_array[0, 2]
        koi_fpflag_ec = features_array[0, 3]
        koi_prad = features_array[0, 4]
        
        # Vectorized score calculation
        score = 100.0
        score -= koi_fpflag_nt * 30
        score -= koi_fpflag_co * 20
        score -= koi_fpflag_ss * 15
        score -= koi_fpflag_ec * 10
        
        # Planet size bonus
        if 0.5 <= koi_prad <= 2.0:
            score += 10
        elif koi_prad > 4.0:
            score -= 5
        
        confidence = max(15.0, min(95.0, score))
        exoplanet_detected = confidence > 50.0
        
        result = {
            'exoplanet_detected': bool(exoplanet_detected),
            'confidence': float(confidence),
            'predictions': [exoplanet_detected],
            'probabilities': [[1-confidence/100, confidence/100]],
            'feature_importance': {
                'koi_fpflag_nt': 0.35,
                'koi_fpflag_co': 0.20,
                'koi_prad': 0.25,
                'koi_fpflag_ss': 0.12,
                'koi_fpflag_ec': 0.08
            },
            'model_type': 'fast_inference'
        }
        
        # Cache result
        self.feature_cache[cache_key] = result
        return result


# Global fast inference engine
fast_engine = FastInferenceEngine()


def performance_monitor():
    """Monitor and log performance metrics."""
    stats = optimizer.get_performance_stats()
    
    logger.info("=== Performance Statistics ===")
    for func_name, metrics in stats.items():
        logger.info(f"{func_name}: avg={metrics['avg_time']:.3f}s, "
                   f"count={metrics['count']}, max={metrics['max_time']:.3f}s")
    
    # Check for performance issues
    for func_name, metrics in stats.items():
        if metrics['avg_time'] > 2.0:
            logger.warning(f"Performance issue: {func_name} average time {metrics['avg_time']:.3f}s > 2s")
        if metrics['max_time'] > 5.0:
            logger.error(f"Critical performance issue: {func_name} max time {metrics['max_time']:.3f}s > 5s")


class PerformanceMiddleware:
    """Flask middleware for performance monitoring."""
    
    def __init__(self, app=None):
        self.app = app
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize performance middleware."""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        
        # Store request start time
        if not hasattr(app, 'request_start_times'):
            app.request_start_times = {}
    
    def before_request(self):
        """Record request start time."""
        import flask
        flask.g.start_time = time.time()
    
    def after_request(self, response):
        """Log request performance."""
        import flask
        
        if hasattr(flask.g, 'start_time'):
            duration = time.time() - flask.g.start_time
            
            # Log slow requests
            if duration > 3.0:
                logger.warning(f"Slow request: {flask.request.endpoint} took {duration:.2f}s")
            
            # Add performance header
            response.headers['X-Response-Time'] = f"{duration:.3f}s"
        
        return response


def optimize_for_production():
    """Apply production optimizations."""
    # Clear any existing caches
    optimizer.clear_cache()
    
    # Set optimal logging level
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # Pre-warm caches with common operations
    logger.info("Pre-warming caches for production...")
    
    # Common feature combinations
    common_features = [
        pd.DataFrame([{'koi_fpflag_nt': 0, 'koi_fpflag_co': 0, 'koi_fpflag_ss': 0, 'koi_fpflag_ec': 0, 'koi_prad': 1.0}]),
        pd.DataFrame([{'koi_fpflag_nt': 1, 'koi_fpflag_co': 0, 'koi_fpflag_ss': 0, 'koi_fpflag_ec': 0, 'koi_prad': 2.0}]),
        pd.DataFrame([{'koi_fpflag_nt': 0, 'koi_fpflag_co': 1, 'koi_fpflag_ss': 0, 'koi_fpflag_ec': 0, 'koi_prad': 0.8}]),
    ]
    
    for features in common_features:
        fast_engine.predict(features)
    
    logger.info("Production optimizations applied")


if __name__ == "__main__":
    # Test performance optimizations
    import pandas as pd
    
    # Test feature extraction
    test_df = pd.DataFrame([{
        'koi_fpflag_nt': 0,
        'koi_fpflag_co': 0,
        'koi_fpflag_ss': 0,
        'koi_fpflag_ec': 0,
        'koi_prad': 1.2
    }])
    
    # Test fast inference
    start_time = time.time()
    result = fast_engine.predict(test_df)
    end_time = time.time()
    
    print(f"Fast inference time: {end_time - start_time:.4f}s")
    print(f"Result: {result}")
    
    # Show performance stats
    performance_monitor()
