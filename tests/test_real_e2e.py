#!/usr/bin/env python3
"""
Real End-to-End Test for LLM RL Framework

This test runs the ACTUAL framework by invoking main.py directly and validates:
1. Framework executes successfully with real LLM and Docker
2. Trajectory files are created with correct structure
3. Results contain actual LLM actions and Docker execution
4. Unit tests run and provide meaningful scores

REQUIREMENTS:
- Docker must be running
- Valid LLM API key must be provided (OPENROUTER_API_KEY or OPENAI_API_KEY)
- Internet connection for LLM API calls
- Sufficient system resources for Docker containers
"""

import asyncio
import json
import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))


class RealE2ETest:
    """Real end-to-end test that invokes main.py directly"""
    
    def __init__(self):
        self.temp_dir = None
        self.test_config_path = None
        self.results_file = None
        self.trajectory_dir = None
        
    def setup_test_environment(self):
        """Set up test environment with config file"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test config based on calculator API example
        self.test_config_path = Path(self.temp_dir) / "test_config.json"
        self.results_file = Path(self.temp_dir) / "results.json"
        self.trajectory_dir = Path(self.temp_dir) / "trajectories"
        
        # Create a simplified test config
        test_config = {
            "environments": [
                {
                    "id": "calculator_api_test",
                    "docker_image": "python:3.9-slim",
                    "init_commands": [
                        "pip install --user flask pytest requests",
                        "echo 'Setup complete'"
                    ],
                    "unit_tests": [
                        "python -c 'import requests; print(\"requests imported successfully\")'",
                        "python -c 'import flask; print(\"flask imported successfully\")'",
                        "python -c 'print(\"Basic Python test passed\")'"
                    ],
                    "prompt": """Create a simple Flask API for a calculator with these requirements:

1. Create a Flask application that provides a calculator API
2. Support basic operations: addition, subtraction, multiplication, division
3. Endpoints:
   - POST /calculate with JSON body: {"operation": "add", "numbers": [1, 2, 3]}
   - GET /health for health check
4. Handle errors gracefully (division by zero, invalid operations)
5. Return results in JSON format: {"result": 6}
6. Run on port 5000

IMPORTANT: After creating the Flask app, you MUST:
1. Start it with 'python app.py &' (run in background)
2. Wait 2-3 seconds with 'sleep 3' 
3. Then mark_done

The unit tests will verify your implementation.""",
                    "working_directory": "/tmp",
                    "environment_variables": {
                        "PYTHONPATH": "/tmp",
                        "FLASK_ENV": "development",
                        "HOME": "/tmp",
                        "PATH": "/tmp/.local/bin:/usr/local/bin:/usr/bin:/bin"
                    }
                }
            ],
            "rollout_config": {
                "max_parallel_rollouts": 1,
                "trajectory_output_path": str(self.trajectory_dir),
                "enable_plugins": False
            },
            "timeout_config": {
                "global_timeout": 300,  # 5 minutes
                "command_timeout": 60,   # 1 minute
                "test_timeout": 30       # 30 seconds
            },
            "template_prompt": "You are an expert software engineer. Complete the task step by step using the available tools.",
            "plugins": []
        }
        
        # Save config to file
        with open(self.test_config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        return self.test_config_path
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def check_prerequisites(self):
        """Check that all prerequisites are met"""
        print("🔍 Checking prerequisites...")
        
        # Check if Docker is running
        try:
            result = subprocess.run(
                ["docker", "info"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                print("✅ Docker is running")
            else:
                print(f"❌ Docker is not running: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Docker check failed: {e}")
            return False
        
        # Check for API key
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ No API key found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY")
            return False
        else:
            print("✅ API key found")
            
        # Check internet connectivity
        try:
            import requests
            response = requests.get("https://httpbin.org/status/200", timeout=5)
            if response.status_code == 200:
                print("✅ Internet connectivity available")
            else:
                print("❌ Internet connectivity issues")
                return False
        except Exception as e:
            print(f"❌ Internet connectivity check failed: {e}")
            return False
        
        return True
    
    def run_main_py(self):
        """Run the actual main.py with real components"""
        print("🚀 Running main.py with real LLM and Docker...")
        
        # Get API key
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # Determine model and base URL
        if api_key and api_key.startswith("sk-or-"):
            model = "anthropic/claude-sonnet-4"
            base_url = "https://openrouter.ai/api/v1"
        else:
            model = "gpt-4"
            base_url = "https://api.openai.com/v1"
        
        # Build command
        cmd = [
            sys.executable, "main.py",
            str(self.test_config_path),
            "--llm-api-key", api_key,
            "--llm-model", model,
            "--llm-base-url", base_url,
            "--llm-temperature", "0.3",
            "--llm-max-tokens", "4000",
            "--output", str(self.results_file),
            "--log-level", "DEBUG"
        ]
        
        print(f"📝 Command: {' '.join(cmd[:3])} [config] [--llm-api-key] [REDACTED] [other args...]")
        print(f"🎯 Model: {model}")
        print(f"🌐 Base URL: {base_url}")
        print(f"📂 Output: {self.results_file}")
        print(f"📁 Trajectories: {self.trajectory_dir}")
        print()
        
        # Run the command
        start_time = datetime.now()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout
                cwd=Path(__file__).parent.parent  # Run from project root
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"⏱️  Execution time: {duration:.2f} seconds")
            print(f"📊 Return code: {result.returncode}")
            
            if result.stdout:
                print(f"📤 STDOUT:\n{result.stdout}")
            
            if result.stderr:
                print(f"📥 STDERR:\n{result.stderr}")
            
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            print("❌ Command timed out after 10 minutes")
            return False, "", "Command timed out"
        except Exception as e:
            print(f"❌ Error running command: {e}")
            return False, "", str(e)
    
    def validate_results(self):
        """Validate that the framework produced correct results"""
        print("🔍 Validating results...")
        
        # Check that paths are set
        if self.results_file is None or self.trajectory_dir is None:
            print("❌ Test paths not initialized properly")
            return False
        
        # Check that results file exists
        if not self.results_file.exists():
            print(f"❌ Results file not found: {self.results_file}")
            return False
        
        print(f"✅ Results file created: {self.results_file}")
        
        # Check that trajectory directory exists
        if not self.trajectory_dir.exists():
            print(f"❌ Trajectory directory not found: {self.trajectory_dir}")
            return False
        
        print(f"✅ Trajectory directory created: {self.trajectory_dir}")
        
        # Check trajectory files
        trajectory_files = list(self.trajectory_dir.glob("*.json"))
        if not trajectory_files:
            print(f"❌ No trajectory files found in {self.trajectory_dir}")
            return False
        
        print(f"✅ Found {len(trajectory_files)} trajectory files")
                    
        # Validate trajectory content
        for traj_file in trajectory_files:
            try:
                with open(traj_file, 'r') as f:
                    traj_data = json.load(f)
                        
                print(f"📝 Trajectory {traj_file.name}:")
                print(f"  - Environment: {traj_data.get('environment_id', 'N/A')}")
                print(f"  - Steps: {len(traj_data.get('steps', []))}")
                print(f"  - Test results: {len(traj_data.get('test_results', []))}")
                        
                # Check required fields
                required_fields = ['trajectory_id', 'environment_id', 'steps']
                for field in required_fields:
                    if field not in traj_data:
                        print(f"❌ Missing required field: {field}")
                        return False
                
                # Check steps structure
                steps = traj_data.get('steps', [])
                if steps:
                    print(f"  - First few actions:")
                    for i, step in enumerate(steps[:3]):
                        action = step.get('action', {})
                        result = step.get('result', {})
                        print(f"    {i+1}. {action.get('type', 'unknown')}: {action.get('content', '')[:50]}...")
                        print(f"       Success: {result.get('success', False)}")
                        
                        # Verify step structure
                        if 'action' not in step or 'result' not in step:
                            print(f"❌ Step {i+1} missing action or result")
                            return False
                
                # Check test results
                test_results = traj_data.get('test_results', [])
                if test_results:
                    passed = sum(1 for t in test_results if t.get('success', False))
                    print(f"  - Tests passed: {passed}/{len(test_results)}")
                
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in trajectory file {traj_file}: {e}")
                return False
        except Exception as e:
                print(f"❌ Error reading trajectory file {traj_file}: {e}")
            return False
        
        # Validate results file
        try:
            with open(self.results_file, 'r') as f:
                results_data = json.load(f)
            
            print(f"📄 Results file validation:")
            if isinstance(results_data, dict):
                print(f"  - Type: dict")
                print(f"  - Keys: {list(results_data.keys())}")
                if 'episodes' in results_data:
                    print(f"  - Episodes: {len(results_data['episodes'])}")
                if 'summary' in results_data:
                    print(f"  - Summary: {bool(results_data['summary'])}")
            else:
                print(f"  - Type: {type(results_data)}")
                print(f"  - Length: {len(results_data) if hasattr(results_data, '__len__') else 'N/A'}")
                    
        except Exception as e:
            print(f"❌ Error reading results file: {e}")
            return False
        
        print("✅ All validation checks passed!")
        return True
    
    async def test_full_real_framework(self):
        """Test the complete framework by running main.py"""
        print("🎯 Testing COMPLETE framework by invoking main.py...")
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("❌ Prerequisites not met. Skipping test.")
            return False
        
        try:
            # Setup test environment
            self.setup_test_environment()
            
            # Run main.py
            success, stdout, stderr = self.run_main_py()
            
            if not success:
                print(f"❌ main.py execution failed")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
            
            # Validate results
            validation_success = self.validate_results()
            
            if validation_success:
                print("✅ REAL end-to-end test PASSED!")
                print("🎉 Framework successfully executed with real LLM + Docker!")
                    return True
            else:
                print("❌ Result validation failed")
                return False
                    
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup_test_environment()
    
    async def run_all_tests(self):
        """Run all tests"""
        print("🚨 REAL END-TO-END TEST - INVOKING main.py 🚨")
        print("=" * 60)
        print("⚠️  This will:")
        print("   - Run the actual main.py with real LLM API calls")
        print("   - Create real Docker containers")
        print("   - Execute LLM-generated commands in containers")
        print("   - Collect actual trajectories")
        print("=" * 60)
        print()
        
        success = await self.test_full_real_framework()
        
        print(f"\n{'='*60}")
        if success:
            print("🎉 REAL END-TO-END TEST PASSED!")
            print("✅ main.py executed successfully with real components!")
        else:
            print("❌ REAL END-TO-END TEST FAILED!")
            print("💥 Check the output above for details")
        print(f"{'='*60}")
        
        return success


async def main():
    """Run the real end-to-end test"""
    print("🚨 REAL END-TO-END TEST MODE 🚨")
    print("This will invoke main.py with real LLM API calls and Docker containers!")
    print()
    
    # Check if user wants to continue
    try:
        response = input("Do you want to continue? (y/N): ")
        if response.lower() != 'y':
            print("Test cancelled by user.")
            return
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
        return
    
    test_suite = RealE2ETest()
    success = await test_suite.run_all_tests()
    
    if success:
        print("\n🎉 Real end-to-end test PASSED!")
        print("The framework works correctly with real LLM API calls and Docker!")
        sys.exit(0)
    else:
        print("\n💥 Real end-to-end test FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 