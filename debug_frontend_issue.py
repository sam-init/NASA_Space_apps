#!/usr/bin/env python3
"""
Debug Frontend Issue
Test exactly what the frontend is doing
"""

import requests
import json

def debug_frontend_workflow():
    """Debug the exact workflow the frontend uses."""
    print("🔍 Debugging Frontend Workflow")
    print("=" * 40)
    
    # Step 1: Upload file (exactly like frontend)
    print("📤 Uploading file...")
    
    with open('sample_exoplanet_data.csv', 'rb') as f:
        files = {'file': ('sample_exoplanet_data.csv', f, 'text/csv')}
        response = requests.post('http://localhost:5000/api/upload', files=files)
    
    print(f"Upload status: {response.status_code}")
    if response.status_code != 200:
        print("❌ Upload failed")
        print("Response:", response.text)
        return
    
    upload_data = response.json()
    file_id = upload_data['file_id']
    print(f"✅ File ID: {file_id}")
    print(f"Validation: {upload_data.get('validation', {})}")
    
    # Step 2: Analyze (exactly like frontend)
    print(f"\n🤖 Analyzing...")
    
    analysis_request = {
        'file_id': file_id,
        'analysis_options': {
            'model_type': 'nasa_pipeline',
            'confidence_threshold': 0.5,
            'include_feature_importance': True
        }
    }
    
    response = requests.post(
        'http://localhost:5000/api/analyze',
        json=analysis_request,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Analysis status: {response.status_code}")
    if response.status_code != 200:
        print("❌ Analysis failed")
        print("Response:", response.text)
        return
    
    analysis_data = response.json()
    print("✅ Analysis successful")
    print(f"Raw response: {json.dumps(analysis_data, indent=2)}")
    
    # Step 3: Show what frontend would display
    print(f"\n📊 Frontend Display:")
    print(f"Detected: {analysis_data.get('exoplanet_detected', 'N/A')}")
    print(f"Confidence: {analysis_data.get('confidence', 'N/A'):.1f}%")
    
    transit_params = analysis_data.get('transit_parameters', {})
    if transit_params:
        depth = transit_params.get('transit_depth_ppm', 0)
        if depth is not None:
            print(f"Transit Depth: {depth / 10000:.3f}%")
        period = transit_params.get('orbital_period_days', 0)
        if period is not None:
            print(f"Period: {period:.2f} days")
        duration = transit_params.get('transit_duration_hours', 0)
        if duration is not None:
            print(f"Duration: {duration:.1f} hours")
        print(f"Planet Type: {transit_params.get('planet_type', 'Unknown')}")
    
    print(f"Processing Time: {analysis_data.get('processing_time', 0):.2f}s")

if __name__ == "__main__":
    debug_frontend_workflow()
