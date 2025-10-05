#!/usr/bin/env python3
"""
Production Logging and Monitoring Setup
NASA Space Apps Challenge 2025 - Team BrainRot

Comprehensive logging, monitoring, and alerting system.
"""

import logging
import logging.handlers
import os
import sys
import time
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
import structlog
from functools import wraps
import threading
from collections import defaultdict, deque


class NASALogger:
    """Production-ready logging system for NASA Exoplanet API."""
    
    def __init__(self, log_level: str = "INFO", log_dir: str = "logs"):
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = log_dir
        self.setup_logging()
        self.metrics = defaultdict(int)
        self.performance_data = defaultdict(list)
        self.error_counts = defaultdict(int)
        
    def setup_logging(self):
        """Set up structured logging with multiple handlers."""
        # Create logs directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Create formatters
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s", '
            '"module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}'
        )
        
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(console_formatter)
        
        # File handlers
        # General application logs
        app_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, 'nasa_api.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        app_handler.setLevel(self.log_level)
        app_handler.setFormatter(json_formatter)
        
        # Error logs
        error_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, 'errors.log'),
            maxBytes=10*1024*1024,
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(json_formatter)
        
        # Performance logs
        perf_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, 'performance.log'),
            maxBytes=10*1024*1024,
            backupCount=5
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(json_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(app_handler)
        root_logger.addHandler(error_handler)
        
        # Configure specific loggers
        perf_logger = logging.getLogger('performance')
        perf_logger.addHandler(perf_handler)
        
        # Suppress noisy third-party loggers
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        
        self.logger = structlog.get_logger("nasa_api")
        self.perf_logger = structlog.get_logger("performance")
        
    def log_request(self, request_info: Dict[str, Any]):
        """Log API request information."""
        self.logger.info("API request", **request_info)
        self.metrics['total_requests'] += 1
        
    def log_response(self, response_info: Dict[str, Any]):
        """Log API response information."""
        self.logger.info("API response", **response_info)
        
        # Track response times
        if 'response_time' in response_info:
            self.performance_data['response_times'].append(response_info['response_time'])
            
        # Track status codes
        if 'status_code' in response_info:
            self.metrics[f"status_{response_info['status_code']}"] += 1
            
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with full context."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.logger.error("Application error", **error_info)
        self.error_counts[type(error).__name__] += 1
        
    def log_performance(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Log performance metrics."""
        perf_info = {
            'operation': operation,
            'duration': duration,
            'metadata': metadata or {}
        }
        
        self.perf_logger.info("Performance metric", **perf_info)
        self.performance_data[operation].append(duration)
        
        # Alert on slow operations
        if duration > 3.0:
            self.logger.warning("Slow operation detected", 
                              operation=operation, 
                              duration=duration)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        metrics = dict(self.metrics)
        
        # Add performance statistics
        for operation, times in self.performance_data.items():
            if times:
                metrics[f"{operation}_avg_time"] = sum(times) / len(times)
                metrics[f"{operation}_max_time"] = max(times)
                metrics[f"{operation}_count"] = len(times)
        
        # Add error statistics
        metrics['error_counts'] = dict(self.error_counts)
        
        return metrics
    
    def reset_metrics(self):
        """Reset metrics (useful for periodic reporting)."""
        self.metrics.clear()
        self.performance_data.clear()
        self.error_counts.clear()


class PerformanceMonitor:
    """Monitor API performance and health."""
    
    def __init__(self, logger: NASALogger):
        self.logger = logger
        self.start_time = time.time()
        self.request_times = deque(maxlen=1000)  # Keep last 1000 requests
        self.error_rate_window = deque(maxlen=100)  # Last 100 requests
        
    def record_request(self, duration: float, success: bool):
        """Record request metrics."""
        self.request_times.append(duration)
        self.error_rate_window.append(not success)
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        now = time.time()
        uptime = now - self.start_time
        
        # Calculate metrics
        if self.request_times:
            avg_response_time = sum(self.request_times) / len(self.request_times)
            max_response_time = max(self.request_times)
            p95_response_time = sorted(self.request_times)[int(len(self.request_times) * 0.95)]
        else:
            avg_response_time = max_response_time = p95_response_time = 0
        
        error_rate = sum(self.error_rate_window) / len(self.error_rate_window) if self.error_rate_window else 0
        
        # Determine health status
        health_status = "healthy"
        if avg_response_time > 2.0 or error_rate > 0.1:
            health_status = "degraded"
        if avg_response_time > 5.0 or error_rate > 0.2:
            health_status = "unhealthy"
        
        return {
            'status': health_status,
            'uptime_seconds': uptime,
            'metrics': {
                'avg_response_time': avg_response_time,
                'max_response_time': max_response_time,
                'p95_response_time': p95_response_time,
                'error_rate': error_rate,
                'total_requests': len(self.request_times)
            },
            'timestamp': datetime.utcnow().isoformat()
        }


def performance_tracker(operation_name: str = None):
    """Decorator to track function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log performance
                if hasattr(wrapper, '_logger'):
                    wrapper._logger.log_performance(operation, duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log error with performance context
                if hasattr(wrapper, '_logger'):
                    wrapper._logger.log_error(e, {
                        'operation': operation,
                        'duration': duration,
                        'args': str(args)[:200],  # Limit arg logging
                        'kwargs': str(kwargs)[:200]
                    })
                
                raise
                
        return wrapper
    return decorator


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, logger: NASALogger):
        self.logger = logger
        self.alert_thresholds = {
            'error_rate': 0.1,  # 10% error rate
            'response_time': 3.0,  # 3 second response time
            'memory_usage': 0.8,  # 80% memory usage
            'disk_usage': 0.9   # 90% disk usage
        }
        
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert thresholds."""
        alerts = []
        
        # Check error rate
        if 'error_rate' in metrics and metrics['error_rate'] > self.alert_thresholds['error_rate']:
            alerts.append({
                'type': 'error_rate',
                'severity': 'high',
                'message': f"High error rate: {metrics['error_rate']:.2%}",
                'threshold': self.alert_thresholds['error_rate']
            })
        
        # Check response time
        if 'avg_response_time' in metrics and metrics['avg_response_time'] > self.alert_thresholds['response_time']:
            alerts.append({
                'type': 'response_time',
                'severity': 'medium',
                'message': f"Slow response time: {metrics['avg_response_time']:.2f}s",
                'threshold': self.alert_thresholds['response_time']
            })
        
        # Log alerts
        for alert in alerts:
            self.logger.logger.warning("Alert triggered", **alert)
        
        return alerts


# Global instances
nasa_logger = NASALogger()
performance_monitor = PerformanceMonitor(nasa_logger)
alert_manager = AlertManager(nasa_logger)


def setup_flask_logging(app):
    """Set up Flask application logging."""
    
    @app.before_request
    def log_request_info():
        """Log incoming request information."""
        import flask
        
        request_info = {
            'method': flask.request.method,
            'url': flask.request.url,
            'remote_addr': flask.request.remote_addr,
            'user_agent': str(flask.request.user_agent),
            'content_length': flask.request.content_length
        }
        
        nasa_logger.log_request(request_info)
        flask.g.start_time = time.time()
    
    @app.after_request
    def log_response_info(response):
        """Log response information."""
        import flask
        
        duration = time.time() - getattr(flask.g, 'start_time', time.time())
        success = 200 <= response.status_code < 400
        
        response_info = {
            'status_code': response.status_code,
            'response_time': duration,
            'content_length': response.content_length
        }
        
        nasa_logger.log_response(response_info)
        performance_monitor.record_request(duration, success)
        
        # Add performance headers
        response.headers['X-Response-Time'] = f"{duration:.3f}s"
        
        return response
    
    @app.errorhandler(Exception)
    def log_exception(error):
        """Log unhandled exceptions."""
        import flask
        
        nasa_logger.log_error(error, {
            'endpoint': flask.request.endpoint,
            'method': flask.request.method,
            'url': flask.request.url
        })
        
        return {'error': 'Internal server error'}, 500
    
    # Add health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint for monitoring."""
        health_status = performance_monitor.get_health_status()
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return health_status, status_code
    
    # Add metrics endpoint
    @app.route('/metrics')
    def metrics_endpoint():
        """Metrics endpoint for monitoring."""
        metrics = nasa_logger.get_metrics()
        health_status = performance_monitor.get_health_status()
        
        return {
            'metrics': metrics,
            'health': health_status,
            'alerts': alert_manager.check_alerts(health_status['metrics'])
        }


if __name__ == "__main__":
    # Test logging setup
    logger = NASALogger()
    
    # Test various log levels
    logger.logger.info("Test info message", test_data="example")
    logger.logger.warning("Test warning message")
    logger.logger.error("Test error message")
    
    # Test performance logging
    logger.log_performance("test_operation", 1.5, {"test": "metadata"})
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.log_error(e, {"context": "test"})
    
    # Show metrics
    print("Metrics:", json.dumps(logger.get_metrics(), indent=2))
