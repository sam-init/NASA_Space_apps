#!/usr/bin/env python3
"""
Test script for NASA Exoplanet Detection Pipeline
NASA Space Apps Challenge 2025 - Team BrainRot

Tests the complete pipeline with NASA Kepler cumulative dataset.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_pipeline_import():
    """Test pipeline import."""
    logger = logging.getLogger(__name__)
    
    try:
        from models.nasa_exoplanet_pipeline import NASAExoplanetPipeline
        logger.info("✅ NASA pipeline import successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Pipeline import failed: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization."""
    logger = logging.getLogger(__name__)
    
    try:
        from models.nasa_exoplanet_pipeline import NASAExoplanetPipeline
        
        pipeline = NASAExoplanetPipeline()
        
        # Check key features
        expected_features = ['koi_fpflag_nt', 'koi_fpflag_co', 'koi_fpflag_ss', 'koi_fpflag_ec', 'koi_prad']
        if pipeline.key_features == expected_features:
            logger.info("✅ Key features correctly configured")
        else:
            logger.warning(f"⚠️ Key features mismatch: {pipeline.key_features}")
        
        # Check dataset URL
        if 'cumulative' in pipeline.dataset_url:
            logger.info("✅ Cumulative dataset URL configured")
        else:
            logger.warning(f"⚠️ Dataset URL: {pipeline.dataset_url}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline initialization failed: {e}")
        return False

def test_dataset_download():
    """Test dataset download (without actually downloading large file)."""
    logger = logging.getLogger(__name__)
    
    try:
        from models.nasa_exoplanet_pipeline import NASAExoplanetPipeline
        import requests
        
        pipeline = NASAExoplanetPipeline()
        
        # Test if URL is accessible
        logger.info("Testing dataset URL accessibility...")
        response = requests.head(pipeline.dataset_url, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Dataset URL is accessible")
            logger.info(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
            return True
        else:
            logger.warning(f"⚠️ Dataset URL returned status: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        logger.warning(f"⚠️ Network test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Dataset download test failed: {e}")
        return False

def test_mock_data_processing():
    """Test data processing with mock data."""
    logger = logging.getLogger(__name__)
    
    try:
        from models.nasa_exoplanet_pipeline import NASAExoplanetPipeline
        
        # Create mock dataset similar to NASA format
        np.random.seed(42)
        n_samples = 1000
        
        mock_data = {
            'koi_disposition': np.random.choice(['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE'], n_samples, p=[0.1, 0.6, 0.3]),
            'koi_fpflag_nt': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'koi_fpflag_co': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'koi_fpflag_ss': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'koi_fpflag_ec': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'koi_prad': np.random.lognormal(0, 1, n_samples)  # Log-normal for planet radii
        }
        
        # Add some erroneous values to koi_fpflag_nt
        mock_data['koi_fpflag_nt'][::100] = -1  # Some erroneous values
        
        mock_df = pd.DataFrame(mock_data)
        
        # Save mock data
        mock_file = "test_mock_cumulative.csv"
        mock_df.to_csv(mock_file, index=False)
        
        # Test processing
        pipeline = NASAExoplanetPipeline()
        pipeline.raw_file = mock_file  # Override to use mock data
        
        X, y = pipeline.load_and_preprocess()
        
        logger.info(f"✅ Mock data processed: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"   Target distribution: {y.value_counts().to_dict()}")
        logger.info(f"   Features: {list(X.columns)}")
        
        # Clean up
        if os.path.exists(mock_file):
            os.remove(mock_file)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Mock data processing failed: {e}")
        return False

def test_model_training():
    """Test model training with mock data."""
    logger = logging.getLogger(__name__)
    
    try:
        from models.nasa_exoplanet_pipeline import NASAExoplanetPipeline
        
        # Create larger mock dataset for training
        np.random.seed(42)
        n_samples = 2000
        
        # Create mock features
        X = pd.DataFrame({
            'koi_fpflag_nt': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'koi_fpflag_co': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'koi_fpflag_ss': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'koi_fpflag_ec': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'koi_prad': np.random.lognormal(0, 1, n_samples)
        })
        
        # Create target with some correlation to features
        y = np.zeros(n_samples)
        # Higher chance of confirmation if flags are 0 and radius is reasonable
        for i in range(n_samples):
            score = 0
            if X.iloc[i]['koi_fpflag_nt'] == 0: score += 1
            if X.iloc[i]['koi_fpflag_co'] == 0: score += 1
            if X.iloc[i]['koi_fpflag_ss'] == 0: score += 1
            if X.iloc[i]['koi_fpflag_ec'] == 0: score += 1
            if 0.5 < X.iloc[i]['koi_prad'] < 10: score += 1
            
            # Probability of confirmation based on score
            prob = min(0.8, score * 0.15)
            y[i] = np.random.choice([0, 1], p=[1-prob, prob])
        
        y = pd.Series(y, dtype=int)
        
        logger.info(f"Mock training data: {X.shape[0]} samples")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Train model
        pipeline = NASAExoplanetPipeline()
        result = pipeline.train_model(X, y)
        
        if result['success']:
            logger.info("✅ Model training successful")
            logger.info(f"   Test Accuracy: {result['metrics']['test_accuracy']:.4f}")
            logger.info(f"   CV Accuracy: {result['cv_scores']['mean']:.4f} ± {result['cv_scores']['std']:.4f}")
            logger.info(f"   Feature Importance: {result['feature_importance']}")
            
            # Test prediction
            sample_X = X.iloc[[0]]
            pred_result = pipeline.predict(sample_X)
            logger.info(f"✅ Prediction test successful: {pred_result}")
            
            return True
        else:
            logger.error(f"❌ Model training failed: {result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Model training test failed: {e}")
        return False

def test_api_integration():
    """Test API integration."""
    logger = logging.getLogger(__name__)
    
    try:
        from app import create_app
        
        app = create_app()
        
        with app.test_client() as client:
            # Test NASA pipeline status
            response = client.get('/api/nasa/pipeline/status')
            if response.status_code == 200:
                data = response.get_json()
                logger.info("✅ NASA pipeline status endpoint working")
                logger.info(f"   Pipeline available: {data.get('pipeline_available', False)}")
                logger.info(f"   Key features: {data.get('key_features', [])}")
            else:
                logger.warning(f"⚠️ Pipeline status endpoint issue: {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API integration test failed: {e}")
        return False

def run_performance_benchmark():
    """Run performance benchmark."""
    logger = logging.getLogger(__name__)
    
    try:
        from models.nasa_exoplanet_pipeline import NASAExoplanetPipeline
        
        logger.info("🚀 Running performance benchmark...")
        
        # Create benchmark dataset
        np.random.seed(42)
        n_samples = 5000
        
        X = pd.DataFrame({
            'koi_fpflag_nt': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'koi_fpflag_co': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'koi_fpflag_ss': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'koi_fpflag_ec': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'koi_prad': np.random.lognormal(0, 1, n_samples)
        })
        
        y = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        y = pd.Series(y, dtype=int)
        
        # Benchmark training
        start_time = datetime.now()
        pipeline = NASAExoplanetPipeline()
        result = pipeline.train_model(X, y)
        training_time = (datetime.now() - start_time).total_seconds()
        
        if result['success']:
            logger.info(f"📊 Benchmark Results:")
            logger.info(f"   Dataset size: {n_samples} samples")
            logger.info(f"   Training time: {training_time:.2f} seconds")
            logger.info(f"   Accuracy: {result['metrics']['test_accuracy']:.4f}")
            logger.info(f"   CV Score: {result['cv_scores']['mean']:.4f}")
            
            # Benchmark prediction
            start_time = datetime.now()
            for _ in range(100):
                pipeline.predict(X.iloc[[0]])
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"   Prediction time (100 samples): {prediction_time:.4f} seconds")
            logger.info(f"   Avg prediction time: {prediction_time/100*1000:.2f} ms")
            
            return True
        else:
            logger.error("❌ Benchmark failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return False

def main():
    """Run all NASA pipeline tests."""
    logger = setup_logging()
    
    print("🚀 NASA Exoplanet Detection Pipeline Test Suite")
    print("=" * 60)
    
    tests = [
        ("Pipeline Import", test_pipeline_import),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Dataset URL Access", test_dataset_download),
        ("Mock Data Processing", test_mock_data_processing),
        ("Model Training", test_model_training),
        ("API Integration", test_api_integration),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 NASA Pipeline Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All NASA pipeline tests passed! Ready for 99%+ accuracy challenge!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
