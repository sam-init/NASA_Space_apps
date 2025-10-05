#!/usr/bin/env python3
"""
Integration Tests for React Frontend
NASA Space Apps Challenge 2025 - Team BrainRot

End-to-end integration tests simulating React frontend interactions.
"""

import unittest
import requests
import json
import time
import os
import tempfile
import threading
import subprocess
import sys
from unittest.mock import patch
import io

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestFrontendIntegration(unittest.TestCase):
    """Integration tests simulating React frontend behavior."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.base_url = "http://localhost:5000"
        cls.api_url = f"{cls.base_url}/api"
        
        # Wait for server to be ready
        cls.wait_for_server()
    
    @classmethod
    def wait_for_server(cls, timeout=30):
        """Wait for the Flask server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(cls.base_url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        
        raise Exception("Server not ready within timeout period")
    
    def test_complete_workflow(self):
        """Test complete frontend workflow: upload -> analyze -> results."""
        # Step 1: Upload file (simulating React file upload)
        nasa_csv_content = """koi_fpflag_nt,koi_fpflag_co,koi_fpflag_ss,koi_fpflag_ec,koi_prad,koi_disposition
0,0,0,0,1.2,CONFIRMED
1,0,0,0,2.5,FALSE POSITIVE
0,1,0,0,0.8,CONFIRMED"""
        
        files = {'file': ('test_data.csv', io.BytesIO(nasa_csv_content.encode('utf-8')), 'text/csv')}
        
        upload_response = requests.post(f"{self.api_url}/upload", files=files)
        self.assertEqual(upload_response.status_code, 200)
        
        upload_data = upload_response.json()
        self.assertIn('file_id', upload_data)
        self.assertTrue(upload_data['ready_for_analysis'])
        
        file_id = upload_data['file_id']
        
        # Step 2: Analyze data (simulating React analysis request)
        analysis_request = {
            'file_id': file_id,
            'analysis_options': {
                'model_type': 'nasa_pipeline',
                'confidence_threshold': 0.5,
                'include_feature_importance': True
            }
        }
        
        start_time = time.time()
        analyze_response = requests.post(
            f"{self.api_url}/analyze",
            json=analysis_request,
            headers={'Content-Type': 'application/json'}
        )
        analysis_time = time.time() - start_time
        
        self.assertEqual(analyze_response.status_code, 200)
        self.assertLess(analysis_time, 3.0, "Analysis should complete in <3s")
        
        analysis_data = analyze_response.json()
        
        # Verify response format matches React expectations
        required_fields = [
            'exoplanet_detected', 'confidence', 'transit_parameters',
            'feature_importance', 'analysis_id', 'processing_time'
        ]
        for field in required_fields:
            self.assertIn(field, analysis_data, f"Missing required field: {field}")
        
        # Verify data types
        self.assertIsInstance(analysis_data['exoplanet_detected'], bool)
        self.assertIsInstance(analysis_data['confidence'], (int, float))
        self.assertIsInstance(analysis_data['transit_parameters'], dict)
        self.assertIsInstance(analysis_data['feature_importance'], dict)
        
        analysis_id = analysis_data['analysis_id']
        
        # Step 3: Get detailed results (simulating React results page)
        results_response = requests.get(f"{self.api_url}/results/{analysis_id}")
        self.assertEqual(results_response.status_code, 200)
        
        results_data = results_response.json()
        self.assertIn('interpretation', results_data)
        self.assertIn('model_performance', results_data)
        
        return {
            'upload': upload_data,
            'analysis': analysis_data,
            'results': results_data
        }
    
    def test_consistency_across_requests(self):
        """Test that multiple requests with same data give consistent results."""
        # Upload same file multiple times
        nasa_csv_content = """koi_fpflag_nt,koi_fpflag_co,koi_fpflag_ss,koi_fpflag_ec,koi_prad
0,0,0,0,1.2"""
        
        results = []
        
        for i in range(3):
            # Upload
            files = {'file': (f'test_data_{i}.csv', io.BytesIO(nasa_csv_content.encode('utf-8')), 'text/csv')}
            upload_response = requests.post(f"{self.api_url}/upload", files=files)
            file_id = upload_response.json()['file_id']
            
            # Analyze
            analysis_request = {
                'file_id': file_id,
                'analysis_options': {
                    'model_type': 'nasa_pipeline',
                    'confidence_threshold': 0.5
                }
            }
            
            analyze_response = requests.post(
                f"{self.api_url}/analyze",
                json=analysis_request,
                headers={'Content-Type': 'application/json'}
            )
            
            analysis_data = analyze_response.json()
            results.append({
                'detected': analysis_data['exoplanet_detected'],
                'confidence': analysis_data['confidence']
            })
        
        # Check consistency
        first_result = results[0]
        for result in results[1:]:
            self.assertEqual(result['detected'], first_result['detected'],
                           "Detection results should be consistent")
            self.assertEqual(result['confidence'], first_result['confidence'],
                           "Confidence scores should be consistent")
    
    def test_error_handling(self):
        """Test error handling scenarios that React frontend might encounter."""
        # Test 1: Invalid file upload
        invalid_files = {'file': ('test.xyz', io.BytesIO(b'invalid content'), 'application/octet-stream')}
        response = requests.post(f"{self.api_url}/upload", files=invalid_files)
        self.assertEqual(response.status_code, 400)
        
        error_data = response.json()
        self.assertIn('error', error_data)
        self.assertIn('supported_formats', error_data)
        
        # Test 2: Analysis with invalid file_id
        invalid_analysis = {
            'file_id': 'invalid-file-id',
            'analysis_options': {'model_type': 'nasa_pipeline'}
        }
        
        response = requests.post(
            f"{self.api_url}/analyze",
            json=invalid_analysis,
            headers={'Content-Type': 'application/json'}
        )
        self.assertEqual(response.status_code, 404)
        
        # Test 3: Results with invalid analysis_id
        response = requests.get(f"{self.api_url}/results/invalid-analysis-id")
        self.assertEqual(response.status_code, 404)
    
    def test_cors_functionality(self):
        """Test CORS headers for React frontend integration."""
        # Test preflight request
        headers = {
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        
        response = requests.options(f"{self.api_url}/upload", headers=headers)
        
        # Check CORS headers
        self.assertIn('Access-Control-Allow-Origin', response.headers)
        self.assertIn('Access-Control-Allow-Methods', response.headers)
        self.assertIn('Access-Control-Allow-Headers', response.headers)
        
        # Verify origin is allowed
        allowed_origin = response.headers.get('Access-Control-Allow-Origin')
        self.assertIn('localhost:3000', allowed_origin)
    
    def test_performance_requirements(self):
        """Test performance requirements for React frontend."""
        # Upload test file
        nasa_csv_content = """koi_fpflag_nt,koi_fpflag_co,koi_fpflag_ss,koi_fpflag_ec,koi_prad
0,0,0,0,1.2"""
        
        files = {'file': ('test_data.csv', io.BytesIO(nasa_csv_content.encode('utf-8')), 'text/csv')}
        upload_response = requests.post(f"{self.api_url}/upload", files=files)
        file_id = upload_response.json()['file_id']
        
        # Test analysis performance
        analysis_request = {
            'file_id': file_id,
            'analysis_options': {'model_type': 'nasa_pipeline'}
        }
        
        # Measure multiple requests
        times = []
        for _ in range(5):
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/analyze",
                json=analysis_request,
                headers={'Content-Type': 'application/json'}
            )
            end_time = time.time()
            
            self.assertEqual(response.status_code, 200)
            times.append(end_time - start_time)
        
        # Check performance requirements
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        self.assertLess(avg_time, 2.0, f"Average response time {avg_time:.2f}s should be <2s")
        self.assertLess(max_time, 3.0, f"Max response time {max_time:.2f}s should be <3s")
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests from React frontend."""
        import concurrent.futures
        
        # Upload test file
        nasa_csv_content = """koi_fpflag_nt,koi_fpflag_co,koi_fpflag_ss,koi_fpflag_ec,koi_prad
0,0,0,0,1.2"""
        
        files = {'file': ('test_data.csv', io.BytesIO(nasa_csv_content.encode('utf-8')), 'text/csv')}
        upload_response = requests.post(f"{self.api_url}/upload", files=files)
        file_id = upload_response.json()['file_id']
        
        def make_analysis_request():
            """Make a single analysis request."""
            analysis_request = {
                'file_id': file_id,
                'analysis_options': {'model_type': 'nasa_pipeline'}
            }
            
            response = requests.post(
                f"{self.api_url}/analyze",
                json=analysis_request,
                headers={'Content-Type': 'application/json'}
            )
            return response.status_code, response.json()
        
        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_analysis_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Check all requests succeeded
        for status_code, data in results:
            self.assertEqual(status_code, 200)
            self.assertIn('confidence', data)
        
        # Check consistency of results
        confidences = [data['confidence'] for _, data in results]
        self.assertEqual(len(set(confidences)), 1, "Concurrent requests should give consistent results")


class TestAPIHealthAndStatus(unittest.TestCase):
    """Test API health and status endpoints for monitoring."""
    
    def setUp(self):
        """Set up test environment."""
        self.base_url = "http://localhost:5000"
        self.api_url = f"{self.base_url}/api"
    
    def test_health_endpoint(self):
        """Test main health endpoint."""
        response = requests.get(self.base_url)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('message', data)
        self.assertIn('timestamp', data)
        self.assertIn('components', data)
    
    def test_api_status(self):
        """Test API status endpoint."""
        response = requests.get(f"{self.api_url}/")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('available_endpoints', data)
        self.assertIsInstance(data['available_endpoints'], dict)
    
    def test_system_status(self):
        """Test system status for monitoring."""
        response = requests.get(f"{self.api_url}/system/status")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('ml_components', data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_nasa_pipeline_status(self):
        """Test NASA pipeline status."""
        response = requests.get(f"{self.api_url}/nasa/pipeline/status")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('pipeline_available', data)
        self.assertIn('key_features', data)
        self.assertTrue(data['pipeline_available'])


class TestDataFormats(unittest.TestCase):
    """Test different data formats supported by the API."""
    
    def setUp(self):
        """Set up test environment."""
        self.api_url = "http://localhost:5000/api"
    
    def test_nasa_koi_format(self):
        """Test NASA KOI CSV format."""
        nasa_csv = """koi_fpflag_nt,koi_fpflag_co,koi_fpflag_ss,koi_fpflag_ec,koi_prad,koi_disposition
0,0,0,0,1.2,CONFIRMED
1,0,0,0,2.5,FALSE POSITIVE"""
        
        files = {'file': ('nasa_koi.csv', io.BytesIO(nasa_csv.encode('utf-8')), 'text/csv')}
        response = requests.post(f"{self.api_url}/upload", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['validation']['nasa_format'])
        self.assertTrue(data['ready_for_analysis'])
    
    def test_json_format(self):
        """Test JSON format upload."""
        json_data = {
            "koi_fpflag_nt": 0,
            "koi_fpflag_co": 0,
            "koi_fpflag_ss": 0,
            "koi_fpflag_ec": 0,
            "koi_prad": 1.2
        }
        
        json_content = json.dumps(json_data)
        files = {'file': ('data.json', io.BytesIO(json_content.encode('utf-8')), 'application/json')}
        response = requests.post(f"{self.api_url}/upload", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['validation']['valid'])
    
    def test_time_series_format(self):
        """Test time series light curve format."""
        time_series_csv = """time,flux,flux_err
0.0,1.0000,0.0001
0.1,0.9999,0.0001
0.2,0.9995,0.0001
0.3,0.9998,0.0001"""
        
        files = {'file': ('lightcurve.csv', io.BytesIO(time_series_csv.encode('utf-8')), 'text/csv')}
        response = requests.post(f"{self.api_url}/upload", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['validation']['valid'])


if __name__ == '__main__':
    # Run integration tests
    unittest.main(verbosity=2, buffer=True)
