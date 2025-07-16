#!/usr/bin/env python3
"""
Test script to verify that the main.py script works with the new CLI parameters.
This test runs the main script with --dry-run to validate configuration loading.
"""

import subprocess
import sys
import tempfile
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_test_config():
    """Create a test configuration file"""
    config = {
        "environments": [
            {
                "id": "test_env",
                "docker_image": "python:3.9-slim",
                "init_commands": ["echo 'test'"],
                "unit_tests": ["python -c \"print('test')\""],
                "prompt": "Simple test prompt",
                "working_directory": "/tmp",
                "environment_variables": {
                    "PYTHONPATH": "/tmp",
                    "HOME": "/tmp",
                    "PATH": "/tmp/.local/bin:/usr/local/bin:/usr/bin:/bin"
                }
            }
        ],
        "rollout_config": {
            "max_parallel_rollouts": 1,
            "trajectory_output_path": "test_trajectories"
        },
        "template_prompt": "Test template",
        "plugins": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        return f.name

def test_main_with_cli_args():
    """Test main.py with CLI arguments"""
    print("🧪 Testing main.py with CLI arguments...")
    
    try:
        # Create test config
        config_file = create_test_config()
        
        # Test with dry-run
        cmd = [
            sys.executable, "main.py",
            config_file,
            "--dry-run",
            "--llm-api-key", "test-key",
            "--llm-model", "anthropic/claude-4-sonnet",
            "--quiet"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        print(f"Exit code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        
        # Clean up
        Path(config_file).unlink()
        
        if result.returncode == 0:
            print("✅ Main script test passed!")
            return True
        else:
            print("❌ Main script test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Main script test failed with exception: {e}")
        return False

def test_help_message():
    """Test help message shows new CLI arguments"""
    print("🧪 Testing help message...")
    
    try:
        cmd = [sys.executable, "main.py", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Check if help message contains new CLI arguments
        help_text = result.stdout
        required_args = [
            "--llm-api-key",
            "--llm-model",
            "--llm-base-url",
            "--llm-temperature",
            "--llm-max-tokens",
            "--llm-timeout"
        ]
        
        missing_args = []
        for arg in required_args:
            if arg not in help_text:
                missing_args.append(arg)
        
        if missing_args:
            print(f"❌ Help message missing arguments: {missing_args}")
            return False
        else:
            print("✅ Help message test passed!")
            return True
            
    except Exception as e:
        print(f"❌ Help message test failed with exception: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing main.py usage with new CLI parameters\n")
    
    tests = [
        test_help_message,
        test_main_with_cli_args
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Main.py Usage Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 