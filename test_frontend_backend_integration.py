#!/usr/bin/env python3
"""
Test Frontend-Backend Integration
Simulates what the frontend does to ensure consistent results
"""

import requests
import json
import time

def test_upload_and_analyze():
    """Test the complete upload and analyze workflow."""
    print("🧪 Testing Frontend-Backend Integration")
    print("=" * 50)
    
    base_url = "http://localhost:5000/api"
    
    # Step 1: Upload file
    print("📤 Step 1: Uploading sample file...")
    
    with open('sample_exoplanet_data.csv', 'rb') as f:
        files = {'file': ('sample_exoplanet_data.csv', f, 'text/csv')}
        upload_response = requests.post(f"{base_url}/upload", files=files)
    
    if upload_response.status_code != 200:
        print(f"❌ Upload failed: {upload_response.status_code}")
        return False
    
    upload_data = upload_response.json()
    file_id = upload_data['file_id']
    print(f"✅ Upload successful: {file_id}")
    print(f"   Format: {upload_data.get('format', 'N/A')}")
    print(f"   NASA format: {upload_data.get('validation', {}).get('nasa_format', False)}")
    
    # Step 2: Analyze multiple times to test consistency
    print(f"\n🤖 Step 2: Testing analysis consistency...")
    
    analysis_request = {
        'file_id': file_id,
        'analysis_options': {
            'model_type': 'nasa_pipeline',
            'confidence_threshold': 0.5,
            'include_feature_importance': True
        }
    }
    
    results = []
    for i in range(3):
        print(f"   Run {i+1}:", end=" ")
        
        analysis_response = requests.post(
            f"{base_url}/analyze",
            json=analysis_request,
            headers={'Content-Type': 'application/json'}
        )
        
        if analysis_response.status_code == 200:
            data = analysis_response.json()
            results.append({
                'detected': data.get('exoplanet_detected', False),
                'confidence': data.get('confidence', 0),
                'processing_time': data.get('processing_time', 0)
            })
            print(f"✅ Detected: {data.get('exoplanet_detected', False)}, Confidence: {data.get('confidence', 0):.1f}%")
        else:
            print(f"❌ Failed ({analysis_response.status_code})")
            return False
        
        time.sleep(0.5)  # Small delay between requests
    
    # Step 3: Check consistency
    print(f"\n📊 Step 3: Checking consistency...")
    
    if len(set(r['detected'] for r in results)) == 1:
        print("✅ Detection results: CONSISTENT")
    else:
        print("❌ Detection results: INCONSISTENT")
        return False
    
    if len(set(f"{r['confidence']:.1f}" for r in results)) == 1:
        print("✅ Confidence scores: CONSISTENT")
    else:
        print("❌ Confidence scores: INCONSISTENT")
        values = [f"{r['confidence']:.1f}%" for r in results]
        print(f"   Values: {values}")
        return False
    
    # Step 4: Show final result
    final_result = results[0]
    print(f"\n🎯 Final Result:")
    print(f"   Exoplanet Detected: {'✅ YES' if final_result['detected'] else '❌ NO'}")
    print(f"   Confidence: {final_result['confidence']:.1f}%")
    print(f"   Processing Time: {final_result['processing_time']:.2f}s")
    
    print(f"\n🎉 SUCCESS: Frontend will now get consistent results!")
    return True

if __name__ == "__main__":
    success = test_upload_and_analyze()
    if success:
        print("\n✅ Integration test PASSED")
        print("🌐 Try the frontend at http://localhost:3000")
        print("📤 Upload sample_exoplanet_data.csv")
        print("🔄 Click 'Start Analysis' multiple times - results will be identical!")
    else:
        print("\n❌ Integration test FAILED")
        print("🔧 Check that backend is running: python backend/app.py")
