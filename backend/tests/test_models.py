#!/usr/bin/env python3
"""
Unit Tests for ML Models
NASA Space Apps Challenge 2025 - Team BrainRot

Tests for NASA exoplanet detection models and pipelines.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.nasa_exoplanet_pipeline import NASAExoplanetPipeline
from models.exoplanet_detector import ExoplanetDetector
from models.data_preprocessor import DataPreprocessor
from models.koi_dataset_loader import KOIDatasetLoader


class TestNASAExoplanetPipeline(unittest.TestCase):
    """Test cases for NASA Exoplanet Pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = NASAExoplanetPipeline()
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'koi_fpflag_nt': [0, 1, 0, 0, 1],
            'koi_fpflag_co': [0, 0, 1, 0, 0],
            'koi_fpflag_ss': [0, 0, 0, 0, 1],
            'koi_fpflag_ec': [0, 1, 0, 0, 0],
            'koi_prad': [1.2, 2.5, 0.8, 3.1, 1.0],
            'koi_disposition': ['CONFIRMED', 'FALSE POSITIVE', 'CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
        })
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsInstance(self.pipeline, NASAExoplanetPipeline)
        self.assertEqual(len(self.pipeline.key_features), 5)
        self.assertIn('koi_fpflag_nt', self.pipeline.key_features)
        self.assertIn('koi_prad', self.pipeline.key_features)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        X, y = self.pipeline.preprocess_data(self.sample_data)
        
        # Check output types
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        
        # Check shapes
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X.columns), 0)
        
        # Check target values are binary
        unique_targets = y.unique()
        self.assertTrue(all(t in [0, 1] for t in unique_targets))
    
    def test_feature_engineering(self):
        """Test feature engineering."""
        X, y = self.pipeline.preprocess_data(self.sample_data)
        
        # Check if engineered features are present
        if 'planet_size_category' in X.columns:
            self.assertTrue(X['planet_size_category'].notna().any())
        
        if 'total_flags' in X.columns:
            self.assertTrue((X['total_flags'] >= 0).all())
    
    def test_model_training_mock(self):
        """Test model training with mock data."""
        X, y = self.pipeline.preprocess_data(self.sample_data)
        
        # Mock the model training to avoid long execution
        with patch.object(self.pipeline, 'model') as mock_model:
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([1, 0, 1, 0, 0])
            mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.4], [0.8, 0.2]])
            
            result = self.pipeline.train_model(X, y)
            
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertIn('metrics', result)
    
    def test_predict_functionality(self):
        """Test prediction functionality."""
        # Create test features
        test_features = pd.DataFrame([{
            'koi_fpflag_nt': 0,
            'koi_fpflag_co': 0,
            'koi_fpflag_ss': 0,
            'koi_fpflag_ec': 0,
            'koi_prad': 1.2
        }])
        
        # Mock model for prediction
        with patch.object(self.pipeline, 'model') as mock_model, \
             patch.object(self.pipeline, 'scaler') as mock_scaler:
            
            mock_scaler.transform.return_value = test_features.values
            mock_model.predict.return_value = np.array([1])
            mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
            
            # Mock load_model to return True
            with patch.object(self.pipeline, 'load_model', return_value=True):
                result = self.pipeline.predict(test_features)
                
                self.assertIsInstance(result, dict)
                self.assertIn('predictions', result)
                self.assertIn('probabilities', result)


class TestExoplanetDetector(unittest.TestCase):
    """Test cases for Exoplanet Detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ExoplanetDetector()
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertIsInstance(self.detector, ExoplanetDetector)
        
        info = self.detector.get_model_info()
        self.assertIsInstance(info, dict)
        self.assertIn('name', info)
        self.assertIn('version', info)
    
    def test_detector_predict_mock(self):
        """Test detector prediction with mock data."""
        test_data = np.random.rand(10, 5)
        
        with patch.object(self.detector, 'model') as mock_model:
            mock_model.predict.return_value = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
            mock_model.predict_proba.return_value = np.random.rand(10, 2)
            
            # Mock load_model
            with patch.object(self.detector, 'load_model', return_value=True):
                result = self.detector.predict(test_data)
                
                self.assertIsInstance(result, dict)
                self.assertIn('predictions', result)


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for Data Preprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsInstance(self.preprocessor, DataPreprocessor)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Create data with missing values
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [np.nan, 2, 3, 4, np.nan],
            'col3': [1, 2, 3, 4, 5]
        })
        
        processed = self.preprocessor.handle_missing_values(data)
        
        self.assertIsInstance(processed, pd.DataFrame)
        # Should have fewer or equal NaN values
        self.assertLessEqual(processed.isna().sum().sum(), data.isna().sum().sum())
    
    def test_normalize_features(self):
        """Test feature normalization."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        normalized = self.preprocessor.normalize_features(data)
        
        self.assertIsInstance(normalized, pd.DataFrame)
        self.assertEqual(normalized.shape, data.shape)
        
        # Check if values are normalized (approximately between -3 and 3 for standard scaling)
        self.assertTrue((normalized.abs() <= 5).all().all())


class TestKOIDatasetLoader(unittest.TestCase):
    """Test cases for KOI Dataset Loader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = KOIDatasetLoader()
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        self.assertIsInstance(self.loader, KOIDatasetLoader)
        
        info = self.loader.get_dataset_info()
        self.assertIsInstance(info, dict)
        self.assertIn('name', info)
    
    def test_load_sample_data(self):
        """Test loading sample data."""
        sample_data = self.loader.load_sample_data(n_samples=10)
        
        self.assertIsInstance(sample_data, pd.DataFrame)
        self.assertEqual(len(sample_data), 10)
        self.assertGreater(len(sample_data.columns), 0)
    
    @patch('requests.get')
    def test_download_dataset_mock(self, mock_get):
        """Test dataset download with mock."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "col1,col2,col3\n1,2,3\n4,5,6\n"
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_dataset.csv")
            
            result = self.loader.download_dataset(save_path=file_path)
            
            self.assertTrue(result)
            if os.path.exists(file_path):
                self.assertTrue(os.path.getsize(file_path) > 0)


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for performance metrics and timing."""
    
    def test_inference_speed(self):
        """Test that inference completes within 3 seconds."""
        import time
        from api.endpoints import generate_deterministic_prediction
        
        # Create test data
        features_df = pd.DataFrame([{
            'koi_fpflag_nt': 0,
            'koi_fpflag_co': 0,
            'koi_fpflag_ss': 0,
            'koi_fpflag_ec': 0,
            'koi_prad': 1.2
        }])
        
        start_time = time.time()
        result = generate_deterministic_prediction(features_df)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete in less than 3 seconds
        self.assertLess(processing_time, 3.0)
        self.assertIsInstance(result, dict)
        self.assertIn('confidence', result)
    
    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        import time
        from api.endpoints import generate_deterministic_prediction
        
        # Create batch of test data
        batch_size = 10
        features_list = []
        
        for i in range(batch_size):
            features_df = pd.DataFrame([{
                'koi_fpflag_nt': i % 2,
                'koi_fpflag_co': (i + 1) % 2,
                'koi_fpflag_ss': 0,
                'koi_fpflag_ec': 0,
                'koi_prad': 1.0 + (i * 0.1)
            }])
            features_list.append(features_df)
        
        start_time = time.time()
        results = []
        for features in features_list:
            result = generate_deterministic_prediction(features)
            results.append(result)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_prediction = total_time / batch_size
        
        # Average time per prediction should be well under 1 second
        self.assertLess(avg_time_per_prediction, 1.0)
        self.assertEqual(len(results), batch_size)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
