#!/usr/bin/env python3
"""
Test script for ExoPlanet AI Backend
NASA Space Apps Challenge 2025 - Team BrainRot

This script tests the backend components to ensure everything is working correctly.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from models.koi_dataset_loader import KOIDatasetLoader
        from models.exoplanet_detector import ExoplanetDetector
        from models.data_preprocessor import DataPreprocessor
        from models.nasa_exoplanet_pipeline import NASAExoplanetPipeline
        from api.endpoints import api_bp
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_flask_app():
    """Test Flask app creation."""
    print("\n🔍 Testing Flask App...")
    
    try:
        from app import create_app
        
        app = create_app()
        
        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/')
            if response.status_code == 200:
                data = response.get_json()
                print(f"✅ Flask app running: {data.get('message', 'N/A')}")
            else:
                print(f"❌ Flask app health check failed: {response.status_code}")
                return False
            
            # Test API root endpoint
            response = client.get('/api/')
            if response.status_code == 200:
                print("✅ API root endpoint working")
            else:
                print(f"⚠️  API root endpoint issue: {response.status_code}")
            
            # Test NASA pipeline status
            response = client.get('/api/nasa/pipeline/status')
            if response.status_code == 200:
                print("✅ NASA pipeline status endpoint working")
            else:
                print(f"⚠️  NASA pipeline status issue: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask App test failed: {e}")
        return False

def test_nasa_pipeline():
    """Test NASA pipeline components."""
    print("\n🔍 Testing NASA Pipeline...")
    
    try:
        from models.nasa_exoplanet_pipeline import NASAExoplanetPipeline
        
        pipeline = NASAExoplanetPipeline()
        
        # Test pipeline initialization
        print(f"✅ Pipeline initialized with {len(pipeline.key_features)} key features")
        print(f"   Features: {pipeline.key_features}")
        
        # Test dataset URL accessibility
        import requests
        try:
            response = requests.head(pipeline.dataset_url, timeout=10)
            if response.status_code == 200:
                print("✅ NASA dataset URL accessible")
            else:
                print(f"⚠️  Dataset URL returned: {response.status_code}")
        except:
            print("⚠️  Dataset URL test failed (network issue)")
        
        return True
        
    except Exception as e:
        print(f"❌ NASA Pipeline test failed: {e}")
        return False

def test_ml_components():
    """Test ML components."""
    print("\n🔍 Testing ML Components...")
    
    try:
        from models.exoplanet_detector import ExoplanetDetector
        from models.data_preprocessor import DataPreprocessor
        from models.koi_dataset_loader import KOIDatasetLoader
        
        # Test ExoplanetDetector
        detector = ExoplanetDetector()
        info = detector.get_model_info()
        print(f"✅ ExoplanetDetector: {info['name']} v{info['version']}")
        
        # Test DataPreprocessor
        preprocessor = DataPreprocessor()
        print("✅ DataPreprocessor initialized")
        
        # Test KOIDatasetLoader
        loader = KOIDatasetLoader()
        dataset_info = loader.get_dataset_info()
        print(f"✅ KOIDatasetLoader: {dataset_info['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Components test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 ExoPlanet AI Backend Test Suite")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    tests = [
        ("Imports", test_imports),
        ("Flask App", test_flask_app),
        ("NASA Pipeline", test_nasa_pipeline),
        ("ML Components", test_ml_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Backend is ready for NASA Space Apps Challenge!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)