#!/usr/bin/env python3
"""
Unit Tests for API Endpoints
NASA Space Apps Challenge 2025 - Team BrainRot

Tests for Flask API endpoints and integration.
"""

import unittest
import json
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock
import io

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import create_app
from api.endpoints import validate_nasa_format, generate_deterministic_prediction
import pandas as pd


class TestAPIEndpoints(unittest.TestCase):
    """Test cases for API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
        self.client = self.app.test_client()
        
        # Create sample test files
        self.create_test_files()
    
    def create_test_files(self):
        """Create test files for upload testing."""
        # NASA KOI format CSV
        self.nasa_csv_content = """koi_fpflag_nt,koi_fpflag_co,koi_fpflag_ss,koi_fpflag_ec,koi_prad,koi_disposition
0,0,0,0,1.2,CONFIRMED
1,0,0,0,2.5,FALSE POSITIVE
0,1,0,0,0.8,CONFIRMED"""
        
        # Invalid CSV
        self.invalid_csv_content = """col1,col2,col3
1,2,3
4,5,6"""
        
        # JSON format
        self.json_content = {
            "koi_fpflag_nt": 0,
            "koi_fpflag_co": 0,
            "koi_fpflag_ss": 0,
            "koi_fpflag_ec": 0,
            "koi_prad": 1.2
        }
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('message', data)
        self.assertIn('ExoPlanet AI Backend', data['message'])
    
    def test_api_root_endpoint(self):
        """Test API root endpoint."""
        response = self.client.get('/api/')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('message', data)
        self.assertIn('available_endpoints', data)
        self.assertIsInstance(data['available_endpoints'], dict)
    
    def test_file_upload_nasa_format(self):
        """Test file upload with NASA format."""
        # Create a file-like object
        file_data = io.BytesIO(self.nasa_csv_content.encode('utf-8'))
        
        response = self.client.post('/api/upload', data={
            'file': (file_data, 'test_nasa.csv')
        })
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('file_id', data)
        self.assertIn('validation', data)
        self.assertTrue(data['validation']['valid'])
        self.assertTrue(data['validation']['nasa_format'])
    
    def test_file_upload_invalid_format(self):
        """Test file upload with invalid format."""
        file_data = io.BytesIO(b"invalid file content")
        
        response = self.client.post('/api/upload', data={
            'file': (file_data, 'test.xyz')
        })
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_file_upload_no_file(self):
        """Test file upload without file."""
        response = self.client.post('/api/upload')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_analyze_endpoint_success(self):
        """Test analysis endpoint with valid file."""
        # First upload a file
        file_data = io.BytesIO(self.nasa_csv_content.encode('utf-8'))
        upload_response = self.client.post('/api/upload', data={
            'file': (file_data, 'test_nasa.csv')
        })
        
        self.assertEqual(upload_response.status_code, 200)
        upload_data = json.loads(upload_response.data)
        file_id = upload_data['file_id']
        
        # Then analyze
        analysis_request = {
            'file_id': file_id,
            'analysis_options': {
                'model_type': 'nasa_pipeline',
                'confidence_threshold': 0.5,
                'include_feature_importance': True
            }
        }
        
        response = self.client.post('/api/analyze', 
                                  data=json.dumps(analysis_request),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check required fields
        self.assertIn('exoplanet_detected', data)
        self.assertIn('confidence', data)
        self.assertIn('transit_parameters', data)
        self.assertIn('feature_importance', data)
        self.assertIn('analysis_id', data)
        
        # Check data types
        self.assertIsInstance(data['exoplanet_detected'], bool)
        self.assertIsInstance(data['confidence'], (int, float))
        self.assertIsInstance(data['transit_parameters'], dict)
        self.assertIsInstance(data['feature_importance'], dict)
    
    def test_analyze_endpoint_missing_file_id(self):
        """Test analysis endpoint without file_id."""
        analysis_request = {
            'analysis_options': {
                'model_type': 'nasa_pipeline'
            }
        }
        
        response = self.client.post('/api/analyze',
                                  data=json.dumps(analysis_request),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_analyze_endpoint_invalid_file_id(self):
        """Test analysis endpoint with invalid file_id."""
        analysis_request = {
            'file_id': 'invalid-file-id',
            'analysis_options': {
                'model_type': 'nasa_pipeline'
            }
        }
        
        response = self.client.post('/api/analyze',
                                  data=json.dumps(analysis_request),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_results_endpoint(self):
        """Test results retrieval endpoint."""
        # First create an analysis
        file_data = io.BytesIO(self.nasa_csv_content.encode('utf-8'))
        upload_response = self.client.post('/api/upload', data={
            'file': (file_data, 'test_nasa.csv')
        })
        upload_data = json.loads(upload_response.data)
        file_id = upload_data['file_id']
        
        analysis_request = {
            'file_id': file_id,
            'analysis_options': {
                'model_type': 'nasa_pipeline'
            }
        }
        
        analyze_response = self.client.post('/api/analyze',
                                          data=json.dumps(analysis_request),
                                          content_type='application/json')
        analyze_data = json.loads(analyze_response.data)
        analysis_id = analyze_data['analysis_id']
        
        # Then retrieve results
        response = self.client.get(f'/api/results/{analysis_id}')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('analysis_id', data)
        self.assertIn('result_type', data)
        self.assertIn('interpretation', data)
    
    def test_results_endpoint_invalid_id(self):
        """Test results endpoint with invalid analysis ID."""
        response = self.client.get('/api/results/invalid-analysis-id')
        
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_nasa_pipeline_status(self):
        """Test NASA pipeline status endpoint."""
        response = self.client.get('/api/nasa/pipeline/status')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('pipeline_available', data)
        self.assertIn('key_features', data)
        self.assertIn('model_specs', data)
    
    def test_system_status(self):
        """Test system status endpoint."""
        response = self.client.get('/api/system/status')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('status', data)
        self.assertIn('ml_components', data)
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.client.options('/api/upload',
                                     headers={
                                         'Origin': 'http://localhost:3000',
                                         'Access-Control-Request-Method': 'POST'
                                     })
        
        # Check CORS headers
        self.assertIn('Access-Control-Allow-Origin', response.headers)
        self.assertIn('Access-Control-Allow-Methods', response.headers)


class TestAPIHelperFunctions(unittest.TestCase):
    """Test cases for API helper functions."""
    
    def test_validate_nasa_format_csv(self):
        """Test NASA format validation for CSV."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("koi_fpflag_nt,koi_fpflag_co,koi_prad,koi_disposition\n")
            f.write("0,0,1.2,CONFIRMED\n")
            temp_file = f.name
        
        try:
            result = validate_nasa_format(temp_file, 'csv')
            
            self.assertIsInstance(result, dict)
            self.assertTrue(result['valid'])
            self.assertTrue(result['nasa_format'])
            self.assertIn('koi_fpflag_nt', result['columns_found'])
        finally:
            os.unlink(temp_file)
    
    def test_validate_nasa_format_json(self):
        """Test NASA format validation for JSON."""
        # Create temporary JSON file
        test_data = {
            "koi_fpflag_nt": 0,
            "koi_fpflag_co": 0,
            "koi_prad": 1.2
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            result = validate_nasa_format(temp_file, 'json')
            
            self.assertIsInstance(result, dict)
            self.assertTrue(result['valid'])
            self.assertIn('koi_fpflag_nt', result['columns_found'])
        finally:
            os.unlink(temp_file)
    
    def test_generate_deterministic_prediction(self):
        """Test deterministic prediction generation."""
        features_df = pd.DataFrame([{
            'koi_fpflag_nt': 0,
            'koi_fpflag_co': 0,
            'koi_fpflag_ss': 0,
            'koi_fpflag_ec': 0,
            'koi_prad': 1.2
        }])
        
        result = generate_deterministic_prediction(features_df)
        
        self.assertIsInstance(result, dict)
        self.assertIn('exoplanet_detected', result)
        self.assertIn('confidence', result)
        self.assertIn('feature_importance', result)
        
        # Test consistency
        result2 = generate_deterministic_prediction(features_df)
        self.assertEqual(result['confidence'], result2['confidence'])
        self.assertEqual(result['exoplanet_detected'], result2['exoplanet_detected'])


class TestAPIPerformance(unittest.TestCase):
    """Test cases for API performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
        self.client = self.app.test_client()
    
    def test_upload_response_time(self):
        """Test upload endpoint response time."""
        import time
        
        file_data = io.BytesIO(b"koi_fpflag_nt,koi_prad\n0,1.2\n")
        
        start_time = time.time()
        response = self.client.post('/api/upload', data={
            'file': (file_data, 'test.csv')
        })
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Upload should complete in under 2 seconds
        self.assertLess(response_time, 2.0)
        self.assertEqual(response.status_code, 200)
    
    def test_analysis_response_time(self):
        """Test analysis endpoint response time (<3s requirement)."""
        import time
        
        # Upload file first
        file_data = io.BytesIO(b"koi_fpflag_nt,koi_fpflag_co,koi_fpflag_ss,koi_fpflag_ec,koi_prad\n0,0,0,0,1.2\n")
        upload_response = self.client.post('/api/upload', data={
            'file': (file_data, 'test.csv')
        })
        upload_data = json.loads(upload_response.data)
        file_id = upload_data['file_id']
        
        # Time the analysis
        analysis_request = {
            'file_id': file_id,
            'analysis_options': {
                'model_type': 'nasa_pipeline'
            }
        }
        
        start_time = time.time()
        response = self.client.post('/api/analyze',
                                  data=json.dumps(analysis_request),
                                  content_type='application/json')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Analysis should complete in under 3 seconds
        self.assertLess(response_time, 3.0)
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('processing_time', data)
        self.assertLess(data['processing_time'], 3.0)


if __name__ == '__main__':
    unittest.main(verbosity=2, buffer=True)
