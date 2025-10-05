#!/usr/bin/env python3
"""
Production Validation Script
NASA Space Apps Challenge 2025 - Team BrainRot
"""

import sys
import os
import time
import pandas as pd

# Add backend to path
sys.path.append('backend')

def validate_production():
    """Validate production readiness."""
    print("🧪 Running Production Validation Tests...")
    print("=" * 50)
    
    # Test imports
    print("📦 Testing imports...")
    try:
        from api.endpoints import generate_deterministic_prediction
        print("✅ API endpoints module imported")
    except ImportError as e:
        print(f"❌ API import error: {e}")
        return False
    
    try:
        from performance.optimizer import fast_engine
        print("✅ Performance optimizer imported")
        fast_available = True
    except ImportError as e:
        print(f"⚠️  Performance optimizer not available: {e}")
        fast_available = False
    
    try:
        from monitoring.logger import nasa_logger
        print("✅ Monitoring logger imported")
    except ImportError as e:
        print(f"⚠️  Monitoring logger not available: {e}")
    
    # Test core functionality
    print("\n🔬 Testing ML Inference...")
    
    features_df = pd.DataFrame([{
        'koi_fpflag_nt': 0,
        'koi_fpflag_co': 0,
        'koi_fpflag_ss': 0,
        'koi_fpflag_ec': 0,
        'koi_prad': 1.2
    }])
    
    # Test deterministic prediction
    start_time = time.time()
    result1 = generate_deterministic_prediction(features_df)
    time1 = time.time() - start_time
    
    start_time = time.time()
    result2 = generate_deterministic_prediction(features_df)
    time2 = time.time() - start_time
    
    print(f"⚡ Inference time: {time1:.4f}s and {time2:.4f}s")
    print(f"🎯 Consistency: {result1['confidence'] == result2['confidence']}")
    print(f"🚀 Performance target (<3s): {max(time1, time2) < 3.0}")
    print(f"📊 Confidence: {result1['confidence']}%")
    print(f"🔍 Detection: {result1['exoplanet_detected']}")
    
    # Test fast engine if available
    if fast_available:
        print("\n⚡ Testing Fast Engine...")
        start_time = time.time()
        fast_result = fast_engine.predict(features_df)
        fast_time = time.time() - start_time
        
        print(f"⚡ Fast engine time: {fast_time:.4f}s")
        print(f"📊 Fast engine confidence: {fast_result['confidence']}%")
    
    # Performance summary
    print("\n📈 Performance Summary:")
    avg_time = (time1 + time2) / 2
    print(f"  Average inference time: {avg_time:.4f}s")
    print(f"  Target compliance: {'✅ PASS' if avg_time < 3.0 else '❌ FAIL'}")
    print(f"  Consistency: {'✅ PERFECT' if result1['confidence'] == result2['confidence'] else '❌ INCONSISTENT'}")
    
    # Overall status
    performance_ok = avg_time < 3.0
    consistency_ok = result1['confidence'] == result2['confidence']
    
    if performance_ok and consistency_ok:
        print("\n🎉 Production validation completed successfully!")
        print("✅ API is ready for production deployment")
        return True
    else:
        print("\n❌ Production validation failed")
        print("⚠️  Review issues before production deployment")
        return False

if __name__ == "__main__":
    success = validate_production()
    sys.exit(0 if success else 1)
