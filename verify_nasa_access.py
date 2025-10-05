#!/usr/bin/env python3
"""
Simple NASA Data Access Verification
NASA Space Apps Challenge 2025 - Team BrainRot
"""

import requests
import pandas as pd
from io import StringIO

def verify_nasa_access():
    """Verify NASA Exoplanet Archive access."""
    print("🌌 Verifying NASA Exoplanet Archive Access")
    print("=" * 50)
    
    # NASA API endpoint
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
    
    try:
        print("📡 Connecting to NASA Exoplanet Archive...")
        response = requests.get(url, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.content):,} bytes")
        
        if response.status_code == 200:
            # Parse first few rows
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data, nrows=5)
            
            print(f"\n📊 Dataset Preview:")
            print(f"Columns: {len(df.columns)}")
            print(f"Sample rows: {len(df)}")
            
            # Check for key columns
            key_columns = ['koi_fpflag_nt', 'koi_fpflag_co', 'koi_fpflag_ss', 'koi_fpflag_ec', 'koi_prad']
            print(f"\n🔍 Key NASA KOI Columns:")
            
            for col in key_columns:
                if col in df.columns:
                    print(f"   ✅ {col}")
                else:
                    print(f"   ❌ {col}")
            
            # Show first row sample
            if len(df) > 0:
                print(f"\n📋 Sample Data (first row):")
                for col in key_columns:
                    if col in df.columns:
                        print(f"   {col}: {df[col].iloc[0]}")
            
            print(f"\n✅ NASA Data Access: SUCCESSFUL")
            print(f"🌌 API can access live NASA KOI data!")
            return True
            
        else:
            print(f"❌ Access failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_curl_equivalent():
    """Test the equivalent of the curl command."""
    print(f"\n🔧 Testing curl equivalent...")
    
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            # Show first 10 lines (equivalent to head)
            lines = response.text.split('\n')[:10]
            
            print("📋 First 10 lines (curl equivalent):")
            for i, line in enumerate(lines, 1):
                print(f"{i:2d}: {line}")
            
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 NASA Data Access Verification")
    print("NASA Space Apps Challenge 2025 - Team BrainRot")
    print("=" * 60)
    
    # Verify access
    access_ok = verify_nasa_access()
    
    # Test curl equivalent
    if access_ok:
        curl_ok = test_curl_equivalent()
    else:
        curl_ok = False
    
    print(f"\n" + "=" * 60)
    print("📊 Verification Summary:")
    print(f"NASA API Access: {'✅ SUCCESS' if access_ok else '❌ FAILED'}")
    print(f"Data Retrieval: {'✅ SUCCESS' if curl_ok else '❌ FAILED'}")
    
    if access_ok and curl_ok:
        print(f"\n🎉 NASA Exoplanet Archive is fully accessible!")
        print(f"✅ Your API can download and process live NASA data")
        print(f"🌌 Ready for real-time exoplanet detection!")
    else:
        print(f"\n⚠️  NASA data access issues detected")
        print(f"💡 Check network connectivity and try again")
    
if __name__ == "__main__":
    main_result = verify_nasa_access()
