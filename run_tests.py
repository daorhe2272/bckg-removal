#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"✅ {description} completed successfully")
        return True

def run_unit_tests():
    """Run unit tests only"""
    cmd = ["python", "-m", "pytest", "tests/test_app.py", "-v"]
    return run_command(cmd, "Unit Tests")

def run_model_tests():
    """Run model integration tests"""
    cmd = ["python", "-m", "pytest", "tests/test_model.py", "-v"]
    return run_command(cmd, "Model Integration Tests")

def run_streamlit_tests():
    """Run Streamlit component tests"""
    cmd = ["python", "-m", "pytest", "tests/test_streamlit_components.py", "-v"]
    return run_command(cmd, "Streamlit Component Tests")

def run_all_tests():
    """Run all tests"""
    cmd = ["python", "-m", "pytest", "tests/", "-v"]
    return run_command(cmd, "All Tests")

def run_tests_with_coverage():
    """Run all tests with coverage report"""
    cmd = ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=term-missing", "--cov-report=html", "-v"]
    return run_command(cmd, "Tests with Coverage")

def run_fast_tests():
    """Run tests excluding slow/model tests"""
    cmd = ["python", "-m", "pytest", "tests/", "-v", "-m", "not slow and not model"]
    return run_command(cmd, "Fast Tests (excluding model tests)")

def run_performance_tests():
    """Run performance tests only"""
    cmd = ["python", "-m", "pytest", "tests/", "-v", "-m", "performance"]
    return run_command(cmd, "Performance Tests")

def lint_code():
    """Run code linting"""
    try:
        cmd = ["python", "-m", "flake8", ".", "--max-line-length=100", "--ignore=E203,W503"]
        return run_command(cmd, "Code Linting (flake8)")
    except FileNotFoundError:
        print("⚠️  flake8 not installed, skipping linting")
        return True

def check_imports():
    """Check if all dependencies are available"""
    required_modules = [
        'pytest', 'numpy', 'PIL', 'cv2', 'onnxruntime', 'streamlit'
    ]
    
    print("\n🔍 Checking required dependencies...")
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - MISSING")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  Missing modules: {', '.join(missing_modules)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required dependencies are available")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test runner for background removal app")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--model", action="store_true", help="Run model tests only")
    parser.add_argument("--streamlit", action="store_true", help="Run Streamlit component tests")
    parser.add_argument("--coverage", action="store_true", help="Run all tests with coverage")
    parser.add_argument("--fast", action="store_true", help="Run fast tests (exclude model tests)")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--lint", action="store_true", help="Run code linting")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if not os.path.exists("tests"):
        print("❌ Could not find tests directory in current location")
        print("Make sure you're running this script from the project root directory")
        sys.exit(1)
    
    src_path = os.path.join(os.getcwd(), "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    success_count = 0
    total_count = 0
    
    if args.check_deps or not any(vars(args).values()):
        total_count += 1
        if check_imports():
            success_count += 1
    
    if args.unit:
        total_count += 1
        if run_unit_tests():
            success_count += 1
    
    if args.model:
        total_count += 1
        if run_model_tests():
            success_count += 1
    
    if args.streamlit:
        total_count += 1
        if run_streamlit_tests():
            success_count += 1
    
    if args.fast:
        total_count += 1
        if run_fast_tests():
            success_count += 1
    
    if args.performance:
        total_count += 1
        if run_performance_tests():
            success_count += 1
    
    if args.coverage:
        total_count += 1
        if run_tests_with_coverage():
            success_count += 1
    
    if args.lint:
        total_count += 1
        if lint_code():
            success_count += 1
    
    if args.all or (not any([args.unit, args.model, args.streamlit, args.coverage, 
                            args.fast, args.performance, args.lint]) and not args.check_deps):
        total_count += 1
        if run_all_tests():
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Completed: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 