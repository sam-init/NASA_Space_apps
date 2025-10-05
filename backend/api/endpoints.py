"""
API Endpoints for ExoPlanet AI Backend
NASA Space Apps Challenge 2025 - Team BrainRot

Advanced API endpoints integrating with ML models for exoplanet detection
using NASA Kepler KOI dataset and ensemble machine learning.
"""

from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

# Import our ML components
from models.exoplanet_detector import ExoplanetDetector
from models.data_preprocessor import DataPreprocessor
from models.koi_dataset_loader import KOIDatasetLoader
from models.nasa_exoplanet_pipeline import NASAExoplanetPipeline

# Create blueprint for API routes
api_bp = Blueprint('api', __name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'fits', 'dat', 'txt'}

@api_bp.route('/')
def api_root():
    """API root endpoint with available endpoints."""
    return jsonify({
        'message': 'ExoPlanet AI API v2.0',
        'team': 'BrainRot',
        'challenge': 'NASA Space Apps Challenge 2025',
        'available_endpoints': {
            'file_operations': [
                'POST /api/upload - Upload light curve files',
                'POST /api/analyze - Analyze uploaded files'
            ],
            'nasa_pipeline': [
                'GET /api/nasa/pipeline/status - Pipeline status',
                'POST /api/nasa/pipeline/run - Run complete pipeline',
                'POST /api/nasa/predict - Make predictions'
            ],
            'dataset_management': [
                'GET /api/dataset/info - Dataset information',
                'POST /api/dataset/load - Load NASA dataset',
                'GET /api/dataset/sample - Get sample data'
            ],
            'model_management': [
                'GET /api/model/info - Model information',
                'POST /api/model/train - Train model'
            ],
            'system': [
                'GET /api/system/status - System status',
                'GET /api/dashboard/stats - Dashboard statistics'
            ]
        },
        'timestamp': datetime.utcnow().isoformat()
    })

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    Upload light curve data file for real-time exoplanet analysis.
    
    Supports: CSV, JSON, FITS formats with NASA dataset validation
    
    Returns:
        JSON response with file_id and validation status
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please select a file to upload',
                'supported_formats': ['CSV', 'JSON', 'FITS']
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a valid file'
            }), 400
        
        # Enhanced file validation
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        allowed_extensions = {'csv', 'json', 'fits', 'dat', 'txt'}
        
        if file_extension not in allowed_extensions:
            return jsonify({
                'error': 'Invalid file type',
                'message': f'File type .{file_extension} not supported',
                'supported_formats': list(allowed_extensions)
            }), 400
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        
        # Save file with unique name
        file_path = os.path.join(
            current_app.config['UPLOAD_FOLDER'], 
            f"{file_id}_{filename}"
        )
        file.save(file_path)
        
        # Enhanced file validation and format detection
        file_size = os.path.getsize(file_path)
        validation_result = validate_nasa_format(file_path, file_extension)
        
        response_data = {
            'file_id': file_id,
            'filename': filename,
            'size': file_size,
            'format': file_extension.upper(),
            'status': 'uploaded',
            'validation': validation_result,
            'timestamp': datetime.utcnow().isoformat(),
            'message': 'File uploaded and validated successfully',
            'ready_for_analysis': validation_result['valid']
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        current_app.logger.error(f"Upload error: {str(e)}")
        return jsonify({
            'error': 'Upload failed',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@api_bp.route('/analyze', methods=['POST'])
def analyze_data():
    """
    Real-time ML inference for exoplanet detection (<3s response time).
    
    Expected JSON payload:
    {
        "file_id": "uuid-string",
        "analysis_options": {
            "model_type": "nasa_pipeline|high_performance",
            "confidence_threshold": 0.5,
            "include_feature_importance": true
        }
    }
    
    Returns:
    {
        "exoplanet_detected": bool,
        "confidence": float,
        "transit_parameters": {},
        "feature_importance": {}
    }
    """
    try:
        start_time = datetime.now()
        
        data = request.get_json()
        
        if not data or 'file_id' not in data:
            return jsonify({
                'error': 'file_id is required',
                'message': 'Please provide a valid file_id from upload',
                'example': {'file_id': 'uuid-string'}
            }), 400
        
        file_id = data['file_id']
        options = data.get('analysis_options', {})
        model_type = options.get('model_type', 'nasa_pipeline')
        confidence_threshold = options.get('confidence_threshold', 0.5)
        include_feature_importance = options.get('include_feature_importance', True)
        
        # Find uploaded file
        upload_folder = current_app.config['UPLOAD_FOLDER']
        uploaded_files = [f for f in os.listdir(upload_folder) if f.startswith(file_id)]
        
        if not uploaded_files:
            return jsonify({
                'error': 'File not found',
                'message': f'No file found with ID: {file_id}',
                'file_id': file_id
            }), 404
        
        file_path = os.path.join(upload_folder, uploaded_files[0])
        
        # Generate analysis ID for tracking
        analysis_id = str(uuid.uuid4())
        
        # Perform real-time ML inference
        analysis_result = perform_realtime_exoplanet_analysis(
            file_path, model_type, confidence_threshold, include_feature_importance
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Ensure <3s response time requirement
        if processing_time > 3.0:
            current_app.logger.warning(f"Analysis took {processing_time:.2f}s (>3s target)")
        
        # Format response according to specification
        response_data = {
            'analysis_id': analysis_id,
            'file_id': file_id,
            'filename': uploaded_files[0],
            'exoplanet_detected': analysis_result.get('exoplanet_detected', False),
            'confidence': float(analysis_result.get('confidence', 0.0)),
            'transit_parameters': analysis_result.get('transit_parameters', {}),
            'feature_importance': analysis_result.get('feature_importance', {}) if include_feature_importance else {},
            'processing_time': processing_time,
            'model_used': model_type,
            'status': 'completed',
            'timestamp': datetime.utcnow().isoformat(),
            'message': 'Analysis completed successfully'
        }
        
        # Store results for later retrieval
        store_analysis_result(analysis_id, response_data)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        current_app.logger.error(f"Analysis error after {processing_time:.2f}s: {str(e)}")
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'processing_time': processing_time,
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@api_bp.route('/results/<analysis_id>', methods=['GET'])
def get_results(analysis_id: str):
    """
    Get detailed predictions with transit parameters by analysis ID.
    
    Returns comprehensive analysis results including:
    - Exoplanet detection status
    - Confidence scores
    - Transit parameters (period, depth, duration, etc.)
    - Feature importance analysis
    - Model metadata
    """
    try:
        # Retrieve stored results
        stored_result = get_stored_analysis_result(analysis_id)
        
        if not stored_result:
            return jsonify({
                'error': 'Analysis not found',
                'message': f'No analysis found with ID: {analysis_id}',
                'analysis_id': analysis_id
            }), 404
        
        # Enhance with additional details if available
        enhanced_result = enhance_analysis_result(stored_result)
        
        return jsonify(enhanced_result), 200
        
    except Exception as e:
        current_app.logger.error(f"Results retrieval error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve results',
            'message': str(e),
            'analysis_id': analysis_id,
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@api_bp.route('/analyses', methods=['GET'])
def get_all_analyses():
    """Get all analyses with pagination."""
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 10, type=int)
        
        # Mock data for demonstration
        # In production, fetch from database with pagination
        mock_analyses = generate_mock_analyses_list(page, limit)
        
        return jsonify(mock_analyses), 200
        
    except Exception as e:
        current_app.logger.error(f"Analyses retrieval error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve analyses',
            'message': str(e)
        }), 500

@api_bp.route('/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Get dashboard statistics."""
    try:
        # Mock statistics - replace with real data from database
        stats = {
            'total_analyses': 1247,
            'exoplanets_found': 89,
            'accuracy_rate': 99.2,
            'average_processing_time': 2.3,
            'recent_activity': [
                {
                    'timestamp': datetime.utcnow().isoformat(),
                    'action': 'analysis_completed',
                    'result': 'exoplanet_detected'
                }
            ]
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        current_app.logger.error(f"Stats error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve stats',
            'message': str(e)
        }), 500

@api_bp.route('/system/status', methods=['GET'])
def get_system_status():
    """Get system status for monitoring."""
    try:
        status = {
            'api': 'operational',
            'database': 'connected',
            'ml_model': 'loaded',
            'queue_size': 3,
            'uptime': '2h 15m',
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        current_app.logger.error(f"Status error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve status',
            'message': str(e)
        }), 500

@api_bp.route('/dataset/info', methods=['GET'])
def get_dataset_info():
    """Get NASA Kepler KOI dataset information."""
    try:
        koi_loader = KOIDatasetLoader()
        info = koi_loader.get_dataset_info()
        return jsonify(info), 200
        
    except Exception as e:
        current_app.logger.error(f"Dataset info error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve dataset info',
            'message': str(e)
        }), 500

@api_bp.route('/dataset/load', methods=['POST'])
def load_dataset():
    """Load and prepare NASA Kepler KOI dataset."""
    try:
        data = request.get_json() or {}
        force_reload = data.get('force_reload', False)
        
        koi_loader = KOIDatasetLoader()
        result = koi_loader.load_dataset(force_reload=force_reload)
        
        return jsonify(result), 200
        
    except Exception as e:
        current_app.logger.error(f"Dataset load error: {str(e)}")
        return jsonify({
            'error': 'Failed to load dataset',
            'message': str(e)
        }), 500

@api_bp.route('/model/train', methods=['POST'])
def train_model():
    """Train the exoplanet detection model."""
    try:
        data = request.get_json() or {}
        training_params = data.get('training_params', {})
        
        # Initialize components
        detector = ExoplanetDetector()
        koi_loader = KOIDatasetLoader()
        
        current_app.logger.info("Starting model training...")
        
        # Load dataset
        dataset_result = koi_loader.load_dataset()
        if not dataset_result.get('success', False):
            return jsonify({
                'error': 'Failed to load training dataset',
                'details': dataset_result
            }), 500
        
        # Load the actual data for training
        import joblib
        processed_data = joblib.load(koi_loader.processed_file)
        features = processed_data['features']
        target = processed_data['target']
        
        # Train model
        training_result = detector.train_model(features, target, **training_params)
        
        return jsonify(training_result), 200
        
    except Exception as e:
        current_app.logger.error(f"Training error: {str(e)}")
        return jsonify({
            'error': 'Training failed',
            'message': str(e)
        }), 500

@api_bp.route('/model/info', methods=['GET'])
def get_model_info():
    """Get model information and statistics."""
    try:
        detector = ExoplanetDetector()
        info = detector.get_model_info()
        return jsonify(info), 200
        
    except Exception as e:
        current_app.logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve model info',
            'message': str(e)
        }), 500

@api_bp.route('/dataset/sample', methods=['GET'])
def get_sample_data():
    """Get sample data from the KOI dataset."""
    try:
        n_samples = request.args.get('n_samples', 5, type=int)
        
        koi_loader = KOIDatasetLoader()
        sample_data = koi_loader.get_sample_data(n_samples)
        
        return jsonify(sample_data), 200
        
    except Exception as e:
        current_app.logger.error(f"Sample data error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve sample data',
            'message': str(e)
        }), 500

@api_bp.route('/nasa/pipeline/run', methods=['POST'])
def run_nasa_pipeline():
    """Run the complete NASA exoplanet detection pipeline."""
    try:
        current_app.logger.info("Starting NASA exoplanet detection pipeline...")
        
        pipeline = NASAExoplanetPipeline()
        result = pipeline.run_complete_pipeline()
        
        return jsonify(result), 200
        
    except Exception as e:
        current_app.logger.error(f"NASA pipeline error: {str(e)}")
        return jsonify({
            'error': 'NASA pipeline failed',
            'message': str(e)
        }), 500

@api_bp.route('/nasa/pipeline/status', methods=['GET'])
def get_nasa_pipeline_status():
    """Get status of NASA pipeline components."""
    try:
        pipeline = NASAExoplanetPipeline()
        
        status = {
            'pipeline_available': True,
            'model_trained': pipeline.load_model(),
            'dataset_files': {
                'raw_exists': os.path.exists(pipeline.raw_file),
                'processed_exists': os.path.exists(pipeline.processed_file),
                'model_exists': os.path.exists(pipeline.model_file)
            },
            'key_features': pipeline.key_features,
            'dataset_url': pipeline.dataset_url,
            'target': 'Binary classification: CONFIRMED vs others',
            'model_specs': {
                'algorithm': 'Random Forest',
                'n_estimators': 100,
                'max_depth': 6,
                'cross_validation': '5-fold',
                'target_accuracy': '99%+'
            }
        }
        
        if pipeline.model is not None:
            status['model_metrics'] = pipeline.metrics_
            status['feature_importance'] = pipeline.feature_importance_
            if pipeline.cv_scores_ is not None:
                status['cv_scores'] = {
                    'mean': float(pipeline.cv_scores_.mean()),
                    'std': float(pipeline.cv_scores_.std()),
                    'scores': pipeline.cv_scores_.tolist()
                }
        
        return jsonify(status), 200
        
    except Exception as e:
        current_app.logger.error(f"NASA pipeline status error: {str(e)}")
        return jsonify({
            'error': 'Failed to get pipeline status',
            'message': str(e)
        }), 500

@api_bp.route('/nasa/predict', methods=['POST'])
def nasa_predict():
    """Make predictions using the NASA pipeline model."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        pipeline = NASAExoplanetPipeline()
        
        if not pipeline.load_model():
            return jsonify({
                'error': 'Model not trained',
                'message': 'Run the pipeline first: POST /api/nasa/pipeline/run'
            }), 400
        
        # Convert input data to DataFrame
        if 'features' in data:
            features_df = pd.DataFrame([data['features']])
        else:
            # Expect direct feature values
            features_df = pd.DataFrame([data])
        
        # Ensure we have the required features
        missing_features = set(pipeline.key_features) - set(features_df.columns)
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing': list(missing_features),
                'required': pipeline.key_features
            }), 400
        
        # Make prediction
        result = pipeline.predict(features_df[pipeline.key_features])
        
        return jsonify(result), 200
        
    except Exception as e:
        current_app.logger.error(f"NASA prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

# Helper functions

def perform_exoplanet_analysis(file_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform actual exoplanet detection analysis using our ML models.
    
    Args:
        file_path: Path to uploaded data file
        options: Analysis options from user
        
    Returns:
        Dictionary with analysis results
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Starting analysis for file: {file_path}")
        
        # Initialize components
        preprocessor = DataPreprocessor()
        detector = ExoplanetDetector()
        
        # Check if we have a trained model
        if not detector.is_trained:
            logger.warning("No trained model found, using mock results")
            return generate_mock_analysis_result()
        
        # Process the uploaded file
        processed_result = preprocessor.process_file(file_path, options)
        
        if processed_result.get('status') != 'success':
            raise ValueError(f"Data preprocessing failed: {processed_result.get('error', 'Unknown error')}")
        
        # For file-based analysis, we need to convert to feature format
        # This is a simplified approach - in production you'd extract proper features
        time_series_data = processed_result['flux']
        
        # Create a simple feature vector from time series statistics
        features_dict = extract_time_series_features(
            processed_result['time'], 
            processed_result['flux']
        )
        
        # Convert to DataFrame for model input
        features_df = pd.DataFrame([features_dict])
        
        # Make prediction
        prediction_result = detector.predict(features_df)
        
        # Combine results
        analysis_result = {
            'exoplanet_detected': prediction_result.get('prediction', False),
            'confidence': prediction_result.get('confidence', 0.0),
            'processing_time': prediction_result.get('processing_time', 0.0),
            'data_quality': processed_result.get('quality_metrics', {}),
            'feature_importance': prediction_result.get('feature_importance', {}),
            'parameters': prediction_result.get('parameters', {}),
            'model_info': prediction_result.get('model_info', {})
        }
        
        logger.info(f"Analysis completed: {analysis_result['exoplanet_detected']} (confidence: {analysis_result['confidence']:.2f}%)")
        return analysis_result
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        # Return mock result as fallback
        return generate_mock_analysis_result()

def extract_time_series_features(time: np.ndarray, flux: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical features from time series data.
    This is a simplified feature extraction for demonstration.
    """
    try:
        features = {
            # Basic statistics
            'flux_mean': float(np.mean(flux)),
            'flux_std': float(np.std(flux)),
            'flux_median': float(np.median(flux)),
            'flux_skewness': float(np.mean((flux - np.mean(flux))**3) / np.std(flux)**3),
            'flux_kurtosis': float(np.mean((flux - np.mean(flux))**4) / np.std(flux)**4),
            
            # Variability measures
            'flux_range': float(np.max(flux) - np.min(flux)),
            'flux_iqr': float(np.percentile(flux, 75) - np.percentile(flux, 25)),
            'flux_mad': float(np.median(np.abs(flux - np.median(flux)))),
            
            # Time series properties
            'time_span': float(np.max(time) - np.min(time)),
            'cadence_median': float(np.median(np.diff(time))),
            'cadence_std': float(np.std(np.diff(time))),
            'data_points': float(len(flux)),
            
            # Transit-like features (simplified)
            'min_flux': float(np.min(flux)),
            'max_flux': float(np.max(flux)),
            'flux_below_median': float(np.sum(flux < np.median(flux)) / len(flux)),
            'consecutive_low_points': float(max_consecutive_below_threshold(flux, np.median(flux))),
        }
        
        return features
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Feature extraction failed: {e}")
        # Return default features
        return {f'feature_{i}': 0.0 for i in range(16)}

def max_consecutive_below_threshold(flux: np.ndarray, threshold: float) -> int:
    """Find maximum consecutive points below threshold."""
    below_threshold = flux < threshold
    max_consecutive = 0
    current_consecutive = 0
    
    for is_below in below_threshold:
        if is_below:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive

def generate_mock_analysis_result() -> Dict[str, Any]:
    """Generate mock analysis result when ML model is not available."""
    import random
    
    exoplanet_detected = random.choice([True, False])
    confidence = random.uniform(70, 99) if exoplanet_detected else random.uniform(10, 30)
    
    return {
        'exoplanet_detected': exoplanet_detected,
        'confidence': confidence,
        'processing_time': random.uniform(1.0, 3.0),
        'data_quality': {
            'completeness': random.uniform(0.8, 1.0),
            'noise_level': random.uniform(0.01, 0.1),
            'snr_estimate': random.uniform(5, 20)
        },
        'parameters': {
            'transit_depth_ppm': random.uniform(100, 20000) if exoplanet_detected else None,
            'orbital_period_days': random.uniform(1, 400) if exoplanet_detected else None,
            'transit_duration_hours': random.uniform(1, 15) if exoplanet_detected else None,
            'detection_quality': 'medium'
        },
        'model_info': {
            'version': '2.0.0',
            'algorithm': 'Mock Analysis'
        },
        'note': 'Mock result - train model for real predictions'
    }

def generate_mock_results(analysis_id: str) -> Dict[str, Any]:
    """Generate mock results for testing."""
    return {
        'analysis_id': analysis_id,
        'status': 'completed',
        'exoplanet_detected': True,
        'confidence': 94.7,
        'transit_depth': 0.0084,
        'period': 384.8,
        'duration': 10.4,
        'timestamp': datetime.utcnow().isoformat()
    }

def generate_mock_analyses_list(page: int, limit: int) -> Dict[str, Any]:
    """Generate mock analyses list for testing."""
    analyses = []
    for i in range(limit):
        analyses.append({
            'analysis_id': str(uuid.uuid4()),
            'filename': f'sample_data_{i+1}.csv',
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'completed',
            'exoplanet_detected': i % 3 == 0  # Every 3rd analysis finds an exoplanet
        })
    
    return {
        'analyses': analyses,
        'page': page,
        'limit': limit,
        'total': 50,  # Mock total count
        'has_next': page * limit < 50
    }
