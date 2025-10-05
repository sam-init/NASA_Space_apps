#!/usr/bin/env python3
"""
Production Deployment Script
NASA Space Apps Challenge 2025 - Team BrainRot

Automated deployment script for NASA Exoplanet API with comprehensive checks.
"""

import os
import sys
import subprocess
import json
import time
import requests
from pathlib import Path


class DeploymentManager:
    """Manage production deployment process."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.backend_dir = self.project_root / "backend"
        self.frontend_dir = self.project_root / "frontend"
        
    def run_command(self, command, cwd=None, check=True):
        """Run shell command with error handling."""
        print(f"🔧 Running: {command}")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                check=check
            )
            if result.stdout:
                print(result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {e}")
            if e.stderr:
                print(f"Error: {e.stderr}")
            if check:
                raise
            return e
    
    def check_prerequisites(self):
        """Check deployment prerequisites."""
        print("🔍 Checking prerequisites...")
        
        checks = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            checks.append(("Python version", True, f"{python_version.major}.{python_version.minor}"))
        else:
            checks.append(("Python version", False, f"{python_version.major}.{python_version.minor} (requires 3.8+)"))
        
        # Check required files
        required_files = [
            "backend/app.py",
            "backend/requirements.txt",
            "backend/requirements-prod.txt",
            "Dockerfile",
            "docker-compose.yml"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            checks.append((f"File: {file_path}", full_path.exists(), str(full_path)))
        
        # Check Docker
        try:
            result = self.run_command("docker --version", check=False)
            docker_available = result.returncode == 0
            checks.append(("Docker", docker_available, "Available" if docker_available else "Not installed"))
        except:
            checks.append(("Docker", False, "Not available"))
        
        # Print results
        all_passed = True
        for check_name, passed, details in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}: {details}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def run_tests(self):
        """Run comprehensive test suite."""
        print("🧪 Running test suite...")
        
        try:
            result = self.run_command("python run_tests.py", cwd=self.backend_dir)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Tests failed: {e}")
            return False
    
    def build_docker_image(self):
        """Build Docker image."""
        print("🐳 Building Docker image...")
        
        try:
            self.run_command("docker build -t nasa-exoplanet-api .")
            print("✅ Docker image built successfully")
            return True
        except Exception as e:
            print(f"❌ Docker build failed: {e}")
            return False
    
    def deploy_local_docker(self):
        """Deploy using local Docker."""
        print("🚀 Deploying with Docker...")
        
        try:
            # Stop existing containers
            self.run_command("docker-compose down", check=False)
            
            # Start services
            self.run_command("docker-compose up -d")
            
            # Wait for services to start
            print("⏳ Waiting for services to start...")
            time.sleep(30)
            
            # Check health
            if self.check_deployment_health():
                print("✅ Docker deployment successful")
                return True
            else:
                print("❌ Docker deployment health check failed")
                return False
                
        except Exception as e:
            print(f"❌ Docker deployment failed: {e}")
            return False
    
    def deploy_render(self):
        """Deploy to Render using render.yaml."""
        print("☁️  Deploying to Render...")
        
        if not (self.project_root / "render.yaml").exists():
            print("❌ render.yaml not found")
            return False
        
        print("📋 Render deployment configuration found")
        print("🔗 To deploy to Render:")
        print("   1. Push code to GitHub repository")
        print("   2. Connect repository to Render")
        print("   3. Render will automatically deploy using render.yaml")
        print("   4. Monitor deployment in Render dashboard")
        
        return True
    
    def check_deployment_health(self, url="http://localhost:5000"):
        """Check deployment health."""
        print(f"🏥 Checking deployment health at {url}...")
        
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{url}/health", timeout=10)
                if response.status_code == 200:
                    health_data = response.json()
                    print(f"✅ Health check passed: {health_data.get('status', 'unknown')}")
                    
                    # Check API endpoints
                    api_response = requests.get(f"{url}/api/", timeout=10)
                    if api_response.status_code == 200:
                        print("✅ API endpoints accessible")
                        return True
                    else:
                        print(f"⚠️  API endpoints issue: {api_response.status_code}")
                        
                else:
                    print(f"⚠️  Health check failed: {response.status_code}")
                    
            except requests.RequestException as e:
                print(f"⏳ Attempt {attempt + 1}/{max_retries}: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(5)
        
        print("❌ Health check failed after all retries")
        return False
    
    def generate_deployment_report(self):
        """Generate deployment report."""
        print("📊 Generating deployment report...")
        
        report = {
            "deployment_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "project": "NASA Exoplanet Detection API",
            "team": "BrainRot",
            "challenge": "NASA Space Apps Challenge 2025",
            "components": {
                "backend": "Flask API with ML pipeline",
                "frontend": "React application",
                "database": "File-based storage",
                "monitoring": "Structured logging and metrics"
            },
            "endpoints": {
                "health": "/health",
                "api_root": "/api/",
                "upload": "/api/upload",
                "analyze": "/api/analyze",
                "results": "/api/results/<id>",
                "nasa_pipeline": "/api/nasa/pipeline/status"
            },
            "performance_targets": {
                "response_time": "<3s",
                "accuracy": "82%+ on NASA data",
                "uptime": "99%+",
                "concurrent_users": "100+"
            }
        }
        
        # Save report
        report_file = self.project_root / "deployment_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Deployment report saved to {report_file}")
        return report
    
    def deploy(self, target="docker"):
        """Main deployment function."""
        print("🚀 NASA Exoplanet API - Production Deployment")
        print("=" * 60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("❌ Prerequisites check failed")
            return False
        
        # Run tests
        if not self.run_tests():
            print("❌ Test suite failed")
            return False
        
        # Deploy based on target
        success = False
        if target == "docker":
            if self.build_docker_image():
                success = self.deploy_local_docker()
        elif target == "render":
            success = self.deploy_render()
        else:
            print(f"❌ Unknown deployment target: {target}")
            return False
        
        # Generate report
        self.generate_deployment_report()
        
        if success:
            print("\n🎉 Deployment completed successfully!")
            print("🌐 API available at: http://localhost:5000")
            print("📊 Health check: http://localhost:5000/health")
            print("📡 API docs: http://localhost:5000/api/")
        else:
            print("\n❌ Deployment failed")
        
        return success


def main():
    """Main deployment script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy NASA Exoplanet API")
    parser.add_argument("--target", choices=["docker", "render"], default="docker",
                       help="Deployment target (default: docker)")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip test suite")
    
    args = parser.parse_args()
    
    deployer = DeploymentManager()
    
    if args.skip_tests:
        print("⚠️  Skipping tests as requested")
        deployer.run_tests = lambda: True
    
    success = deployer.deploy(args.target)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
