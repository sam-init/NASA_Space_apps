"""
ExoPlanet AI Backend - Flask Application
NASA Space Apps Challenge 2025 - Team BrainRot

Main Flask application with CORS support for React frontend integration.
"""

from flask import Flask, jsonify
from flask_cors import CORS
import logging
import os
from datetime import datetime

def create_app():
    """Create and configure Flask application."""
    
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'nasa-space-apps-2025-brainrot')
    app.config['UPLOAD_FOLDER'] = 'data/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
    
    # Enable CORS for React frontend
    CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])
    
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Register blueprints
    try:
        from api.endpoints import api_bp
        app.register_blueprint(api_bp, url_prefix='/api')
        app.logger.info("API blueprint registered successfully")
    except ImportError as e:
        app.logger.error(f"Failed to import API blueprint: {e}")
    
    # Initialize ML models with proper integration
    try:
        from models.koi_dataset_loader import KOIDatasetLoader
        from models.exoplanet_detector import ExoplanetDetector
        from models.data_preprocessor import DataPreprocessor
        from models.nasa_exoplanet_pipeline import NASAExoplanetPipeline
        
        app.detector = ExoplanetDetector()
        app.preprocessor = DataPreprocessor()
        app.koi_loader = KOIDatasetLoader()
        app.nasa_pipeline = NASAExoplanetPipeline()
        app.logger.info("ML components initialized successfully")
    except Exception as e:
        app.logger.error(f"Failed to initialize ML components: {e}")
        app.detector = None
        app.preprocessor = None
        app.koi_loader = None
        app.nasa_pipeline = None
    
    @app.route('/')
    def index():
        """Health check endpoint."""
        return jsonify({
            'message': 'ExoPlanet AI Backend is running!',
            'version': '2.0.0',
            'team': 'BrainRot',
            'challenge': 'NASA Space Apps Challenge 2025',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'detector': app.detector is not None,
                'preprocessor': app.preprocessor is not None,
                'koi_loader': app.koi_loader is not None,
                'nasa_pipeline': app.nasa_pipeline is not None
            }
        })
    
    @app.route('/health')
    def health():
        """Detailed health check."""
        try:
            # Test ML components
            ml_status = {
                'detector_available': app.detector is not None,
                'preprocessor_available': app.preprocessor is not None,
                'koi_loader_available': app.koi_loader is not None,
                'nasa_pipeline_available': app.nasa_pipeline is not None
            }
            
            # Test NASA pipeline specifically
            nasa_status = 'unknown'
            if app.nasa_pipeline is not None:
                try:
                    # Quick test of pipeline components
                    pipeline_test = app.nasa_pipeline.key_features is not None
                    nasa_status = 'ready' if pipeline_test else 'error'
                except:
                    nasa_status = 'error'
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'ml_components': ml_status,
                'nasa_pipeline_status': nasa_status,
                'upload_folder': app.config['UPLOAD_FOLDER'],
                'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
            })
            
        except Exception as e:
            app.logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 500
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    print("🚀 Starting ExoPlanet AI Backend...")
    print("🌐 Server: http://localhost:5000")
    print("📊 Health: http://localhost:5000/health")
    print("📡 API: http://localhost:5000/api/")
    print("🌌 Ready for NASA Space Apps Challenge!")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )