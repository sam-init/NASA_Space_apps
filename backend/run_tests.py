#!/usr/bin/env python3
"""
Comprehensive Test Runner for NASA Exoplanet API
NASA Space Apps Challenge 2025 - Team BrainRot

Runs all tests with coverage reporting and performance benchmarks.
"""

import unittest
import sys
import os
import time
import json
from io import StringIO
import coverage
import subprocess

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))


class TestRunner:
    """Comprehensive test runner with reporting."""
    
    def __init__(self):
        self.cov = coverage.Coverage()
        self.results = {
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'coverage': {},
            'summary': {}
        }
    
    def run_unit_tests(self):
        """Run unit tests for models and API."""
        print("🧪 Running Unit Tests...")
        
        # Start coverage
        self.cov.start()
        
        # Discover and run unit tests
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Load model tests
        try:
            model_tests = loader.discover('tests', pattern='test_models.py')
            suite.addTests(model_tests)
        except Exception as e:
            print(f"Warning: Could not load model tests: {e}")
        
        # Load API tests
        try:
            api_tests = loader.discover('tests', pattern='test_api.py')
            suite.addTests(api_tests)
        except Exception as e:
            print(f"Warning: Could not load API tests: {e}")
        
        # Run tests
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        result = runner.run(suite)
        
        # Stop coverage
        self.cov.stop()
        
        # Store results
        self.results['unit_tests'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1),
            'output': stream.getvalue()
        }
        
        print(f"✅ Unit Tests: {result.testsRun} run, {len(result.failures)} failures, {len(result.errors)} errors")
        return result.wasSuccessful()
    
    def run_integration_tests(self):
        """Run integration tests."""
        print("🔗 Running Integration Tests...")
        
        # Check if server is running
        if not self.check_server_running():
            print("⚠️  Server not running, skipping integration tests")
            return False
        
        # Load integration tests
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        try:
            integration_tests = loader.discover('tests', pattern='test_integration.py')
            suite.addTests(integration_tests)
        except Exception as e:
            print(f"Warning: Could not load integration tests: {e}")
            return False
        
        # Run tests
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        result = runner.run(suite)
        
        # Store results
        self.results['integration_tests'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1),
            'output': stream.getvalue()
        }
        
        print(f"✅ Integration Tests: {result.testsRun} run, {len(result.failures)} failures, {len(result.errors)} errors")
        return result.wasSuccessful()
    
    def run_performance_tests(self):
        """Run performance benchmarks."""
        print("⚡ Running Performance Tests...")
        
        performance_results = {}
        
        # Test 1: API response time
        try:
            import requests
            import time
            
            times = []
            for i in range(10):
                start = time.time()
                response = requests.get('http://localhost:5000/health', timeout=5)
                end = time.time()
                
                if response.status_code == 200:
                    times.append(end - start)
            
            if times:
                performance_results['api_response_time'] = {
                    'avg': sum(times) / len(times),
                    'max': max(times),
                    'min': min(times),
                    'target': 1.0,  # 1 second target
                    'passed': max(times) < 1.0
                }
        except Exception as e:
            print(f"Performance test failed: {e}")
            performance_results['api_response_time'] = {'error': str(e)}
        
        # Test 2: ML inference speed
        try:
            from api.endpoints import generate_deterministic_prediction
            import pandas as pd
            
            features_df = pd.DataFrame([{
                'koi_fpflag_nt': 0,
                'koi_fpflag_co': 0,
                'koi_fpflag_ss': 0,
                'koi_fpflag_ec': 0,
                'koi_prad': 1.2
            }])
            
            times = []
            for i in range(100):
                start = time.time()
                result = generate_deterministic_prediction(features_df)
                end = time.time()
                times.append(end - start)
            
            performance_results['ml_inference_time'] = {
                'avg': sum(times) / len(times),
                'max': max(times),
                'min': min(times),
                'target': 0.1,  # 100ms target
                'passed': max(times) < 0.1
            }
        except Exception as e:
            performance_results['ml_inference_time'] = {'error': str(e)}
        
        self.results['performance_tests'] = performance_results
        
        # Print results
        for test_name, results in performance_results.items():
            if 'error' in results:
                print(f"❌ {test_name}: {results['error']}")
            else:
                status = "✅" if results['passed'] else "⚠️"
                print(f"{status} {test_name}: avg={results['avg']:.3f}s, max={results['max']:.3f}s (target: {results['target']}s)")
        
        return all(r.get('passed', False) for r in performance_results.values() if 'error' not in r)
    
    def generate_coverage_report(self):
        """Generate coverage report."""
        print("📊 Generating Coverage Report...")
        
        try:
            # Save coverage data
            self.cov.save()
            
            # Generate report
            stream = StringIO()
            self.cov.report(file=stream)
            coverage_text = stream.getvalue()
            
            # Get coverage percentage
            total_coverage = self.cov.report()
            
            self.results['coverage'] = {
                'total_coverage': total_coverage,
                'report': coverage_text,
                'target': 80.0,
                'passed': total_coverage >= 80.0
            }
            
            print(f"📈 Coverage: {total_coverage:.1f}% (target: 80%)")
            
            # Generate HTML report
            try:
                self.cov.html_report(directory='htmlcov')
                print("📄 HTML coverage report generated in htmlcov/")
            except Exception as e:
                print(f"Could not generate HTML report: {e}")
            
        except Exception as e:
            print(f"Coverage report failed: {e}")
            self.results['coverage'] = {'error': str(e)}
    
    def check_server_running(self):
        """Check if the Flask server is running."""
        try:
            import requests
            response = requests.get('http://localhost:5000/health', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_summary(self):
        """Generate test summary."""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': 0,
            'total_failures': 0,
            'total_errors': 0,
            'overall_success_rate': 0,
            'performance_passed': False,
            'coverage_passed': False,
            'production_ready': False
        }
        
        # Aggregate test results
        for test_type in ['unit_tests', 'integration_tests']:
            if test_type in self.results and 'tests_run' in self.results[test_type]:
                summary['total_tests'] += self.results[test_type]['tests_run']
                summary['total_failures'] += self.results[test_type]['failures']
                summary['total_errors'] += self.results[test_type]['errors']
        
        # Calculate overall success rate
        if summary['total_tests'] > 0:
            successful_tests = summary['total_tests'] - summary['total_failures'] - summary['total_errors']
            summary['overall_success_rate'] = successful_tests / summary['total_tests']
        
        # Check performance
        if 'performance_tests' in self.results:
            perf_results = self.results['performance_tests']
            summary['performance_passed'] = all(
                r.get('passed', False) for r in perf_results.values() if 'error' not in r
            )
        
        # Check coverage
        if 'coverage' in self.results and 'passed' in self.results['coverage']:
            summary['coverage_passed'] = self.results['coverage']['passed']
        
        # Determine if production ready
        summary['production_ready'] = (
            summary['overall_success_rate'] >= 0.95 and
            summary['performance_passed'] and
            summary['coverage_passed']
        )
        
        self.results['summary'] = summary
        return summary
    
    def save_results(self, filename='test_results.json'):
        """Save test results to file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Test results saved to {filename}")
    
    def run_all_tests(self):
        """Run all tests and generate reports."""
        print("🚀 NASA Exoplanet API - Comprehensive Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run tests
        unit_success = self.run_unit_tests()
        integration_success = self.run_integration_tests()
        performance_success = self.run_performance_tests()
        
        # Generate reports
        self.generate_coverage_report()
        summary = self.generate_summary()
        
        # Save results
        self.save_results()
        
        total_time = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Failures: {summary['total_failures']}")
        print(f"Errors: {summary['total_errors']}")
        print(f"Success Rate: {summary['overall_success_rate']:.1%}")
        print(f"Performance: {'✅ PASS' if summary['performance_passed'] else '❌ FAIL'}")
        print(f"Coverage: {'✅ PASS' if summary['coverage_passed'] else '❌ FAIL'}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Production Ready: {'✅ YES' if summary['production_ready'] else '❌ NO'}")
        
        if summary['production_ready']:
            print("\n🎉 All tests passed! API is production ready!")
        else:
            print("\n⚠️  Some tests failed. Review results before production deployment.")
        
        return summary['production_ready']


def main():
    """Main test runner entry point."""
    runner = TestRunner()
    success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
