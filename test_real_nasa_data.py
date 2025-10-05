#!/usr/bin/env python3
"""
Test Real NASA Dataset Processing
"""

import requests
import time

def test_real_nasa_data():
    print("🧪 Testing Real NASA Dataset Processing")
    print("=" * 50)
    
    # Upload the real NASA cumulative dataset
    print("📤 Uploading NASA cumulative dataset...")
    with open("cumulative_2025.10.05_02.06.21.csv", "rb") as f:
        files = {"file": ("cumulative_2025.10.05_02.06.21.csv", f, "text/csv")}
        upload_response = requests.post("http://localhost:5000/api/upload", files=files)
    
    if upload_response.status_code == 200:
        upload_data = upload_response.json()
        file_id = upload_data["file_id"]
        print(f"✅ Upload successful: {file_id}")
        
        # Analyze the real NASA data
        print("🤖 Analyzing real NASA dataset...")
        analysis_request = {
            "file_id": file_id,
            "analysis_options": {
                "model_type": "nasa_pipeline",
                "confidence_threshold": 0.5,
                "include_feature_importance": True
            }
        }
        
        start_time = time.time()
        analyze_response = requests.post(
            "http://localhost:5000/api/analyze",
            json=analysis_request,
            headers={"Content-Type": "application/json"}
        )
        analysis_time = time.time() - start_time
        
        if analyze_response.status_code == 200:
            analysis_data = analyze_response.json()
            
            print(f"✅ Analysis completed in {analysis_time:.3f}s")
            print(f"\n📊 Real NASA Data Results:")
            print(f"   Exoplanet Detected: {analysis_data.get('exoplanet_detected', 'N/A')}")
            print(f"   Confidence: {analysis_data.get('confidence', 'N/A')}%")
            
            if "transit_parameters" in analysis_data:
                transit = analysis_data["transit_parameters"]
                print(f"\n🌌 Real Transit Parameters:")
                print(f"   Orbital Period: {transit.get('orbital_period_days', 'N/A')} days")
                print(f"   Transit Depth: {transit.get('transit_depth_ppm', 'N/A')} ppm")
                print(f"   Transit Duration: {transit.get('transit_duration_hours', 'N/A')} hours")
                print(f"   Planet Radius: {transit.get('planet_radius_earth_radii', 'N/A')} Earth radii")
                print(f"   Planet Type: {transit.get('planet_type', 'N/A')}")
                print(f"   Habitable Zone: {transit.get('habitable_zone', 'N/A')}")
                print(f"   Temperature: {transit.get('equilibrium_temperature_k', 'N/A')} K")
            
            print(f"\n🎉 Now processing REAL NASA data instead of dummy values!")
            return True
            
        else:
            print(f"❌ Analysis failed: {analyze_response.status_code}")
            print(f"Response: {analyze_response.text}")
            return False
    else:
        print(f"❌ Upload failed: {upload_response.status_code}")
        print(f"Response: {upload_response.text}")
        return False

if __name__ == "__main__":
    test_real_nasa_data()
