"""
ExoPlanet AI - Flask Backend
NASA Space Apps Challenge 2025 - Team BrainRot

Main Flask application for exoplanet detection using AI/ML models.
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Import custom modules (to be created by your team)
from api.endpoints import api_bp
from models.exoplanet_detector import ExoplanetDetector
from models.data_preprocessor import DataPreprocessor

def create_app(config_name='development'):
    """Application factory pattern for Flask app creation."""
    
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'data', 'uploads')
    app.config['PROCESSED_FOLDER'] = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    
    # Ensure upload directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    
    # Enable CORS for frontend integration
    CORS(app, origins=['http://localhost:3000', 'https://exoplanet-ai.netlify.app'])
    
    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Initialize ML models (placeholder - your team will implement)
    app.detector = ExoplanetDetector()
    app.preprocessor = DataPreprocessor()
    
    @app.route('/')
    def index():
        """Health check endpoint."""
        return jsonify({
            'message': 'ExoPlanet AI Backend is running!',
            'version': '1.0.0',
            'team': 'BrainRot',
            'challenge': 'NASA Space Apps Challenge 2025',
            'timestamp': datetime.utcnow().isoformat()
        })
    
    @app.route('/health')
    def health_check():
        """Detailed health check for monitoring."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {
                'api': 'operational',
                'ml_model': 'loaded' if hasattr(app, 'detector') else 'not_loaded',
                'preprocessor': 'loaded' if hasattr(app, 'preprocessor') else 'not_loaded'
            }
        })
    
    @app.errorhandler(413)
    def too_large(e):
        """Handle file too large errors."""
        return jsonify({
            'error': 'File too large',
            'message': 'Maximum file size is 100MB'
        }), 413
    
    @app.errorhandler(404)
    def not_found(e):
        """Handle 404 errors."""
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource was not found'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        """Handle internal server errors."""
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    # Development server configuration
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"""
    🚀 ExoPlanet AI Backend Starting...
    
    Team: BrainRot
    Challenge: NASA Space Apps Challenge 2025
    Port: {port}
    Debug: {debug}
    
    Frontend should connect to: http://localhost:{port}/api
    Health check: http://localhost:{port}/health
    """)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
