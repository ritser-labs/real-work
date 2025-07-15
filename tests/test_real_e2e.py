#!/usr/bin/env python3
"""
Real End-to-End Test for LLM RL Framework

This test runs the complete framework pipeline:
1. Loads the calculator API example configuration
2. Injects LLM configuration
3. Runs the full rollout manager with mocked LLM responses
4. Collects actual trajectories
5. Verifies output files and trajectory data
6. Tests the complete workflow from start to finish
"""

import asyncio
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import FrameworkConfig, LLMConfig, Action, ActionType, ActionResult, EpisodeResult, TestResult
from src.core.rollout_manager import RolloutManager
from src.agents.llm_agent import LLMAgent
from src.environments.environment import Environment


class MockLLMAgent:
    """Mock LLM agent that provides realistic responses for calculator API task"""
    
    def __init__(self, config):
        self.config = config
        self.conversation_history = []
        self.step_count = 0
        
    async def initialize_conversation(self, template_prompt, environment_prompt):
        """Initialize the conversation"""
        self.conversation_history = [
            {"role": "system", "content": template_prompt},
            {"role": "user", "content": environment_prompt}
        ]
        
    async def get_next_action(self, context=None):
        """Return realistic actions for the calculator API task"""
        self.step_count += 1
        
        if self.step_count == 1:
            # First action: create the Flask app
            return [Action(
                type=ActionType.FILE_WRITE,
                content="""# Writing Flask calculator API
cat > app.py << 'EOF'
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        operation = data.get('operation')
        numbers = data.get('numbers', [])
        
        if not numbers:
            return jsonify({"error": "No numbers provided"}), 400
            
        if operation == 'add':
            result = sum(numbers)
        elif operation == 'subtract':
            result = numbers[0] - sum(numbers[1:])
        elif operation == 'multiply':
            result = 1
            for num in numbers:
                result *= num
        elif operation == 'divide':
            result = numbers[0]
            for num in numbers[1:]:
                if num == 0:
                    return jsonify({"error": "Division by zero"}), 400
                result /= num
        else:
            return jsonify({"error": "Invalid operation"}), 400
            
        return jsonify({"result": result}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
EOF"""
            )]
            
        elif self.step_count == 2:
            # Second action: start the Flask app
            return [Action(
                type=ActionType.COMMAND,
                content="python app.py &"
            )]
            
        elif self.step_count == 3:
            # Third action: wait for app to start
            return [Action(
                type=ActionType.COMMAND,
                content="sleep 2"
            )]
            
        elif self.step_count == 4:
            # Fourth action: mark done
            return [Action(
                type=ActionType.DONE,
                content="Flask calculator API is running and ready for testing"
            )]
            
        else:
            # No more actions
            return []


class MockEnvironment:
    """Mock environment that simulates successful operations"""
    
    def __init__(self, config, timeout_config=None, state_manager=None):
        self.config = config
        self.timeout_config = timeout_config
        self.state_manager = state_manager
        self.initialized = False
        
    async def initialize(self):
        """Mock initialization that always succeeds"""
        self.initialized = True
        return True
    
    async def cleanup(self):
        """Mock cleanup"""
        pass
    
    async def execute_action(self, action):
        """Mock action execution that returns realistic results"""
        if action.type == ActionType.FILE_WRITE:
            return ActionResult(
                success=True,
                output="File written successfully",
                error="",
                exit_code=0,
                duration=0.1,
                timestamp="2024-01-01T00:00:00"
            )
        elif action.type == ActionType.COMMAND:
            return ActionResult(
                success=True,
                output="Command executed successfully",
                error="",
                exit_code=0,
                duration=0.5,
                timestamp="2024-01-01T00:00:00"
            )
        elif action.type == ActionType.DONE:
            return ActionResult(
                success=True,
                output="Task completed",
                error="",
                exit_code=0,
                duration=0.0,
                timestamp="2024-01-01T00:00:00"
            )
        else:
            return ActionResult(
                success=False,
                output="",
                error="Unknown action type",
                exit_code=1,
                duration=0.0,
                timestamp="2024-01-01T00:00:00"
            )
    
    async def run_tests(self):
        """Mock test execution that returns realistic test results"""
        return [
            TestResult(
                command="python -c \"print('Hello, World!')\"",
                success=True,
                output="Hello, World!\n",
                error="",
                exit_code=0,
                duration=0.1,
                timestamp="2024-01-01T00:00:00"
            ),
            TestResult(
                command="python -c \"import requests; print('requests imported successfully')\"",
                success=True,
                output="requests imported successfully\n",
                error="",
                exit_code=0,
                duration=0.2,
                timestamp="2024-01-01T00:00:00"
            )
        ]


class MockTestRunner:
    """Mock test runner that returns realistic test results"""
    
    def __init__(self, timeout_config=None):
        self.timeout_config = timeout_config
    
    async def run_tests(self, environment, test_commands=None):
        """Mock test execution"""
        if test_commands is None:
            test_commands = environment.config.unit_tests
        
        return [
            TestResult(
                command=cmd,
                success=True,
                output=f"Test passed: {cmd}",
                error="",
                exit_code=0,
                duration=0.1,
                timestamp="2024-01-01T00:00:00"
            )
            for cmd in test_commands
        ]
    
    def calculate_test_score(self, test_results):
        """Calculate test score"""
        if not test_results:
            return 0.0
        
        passed = sum(1 for test in test_results if test.success)
        return passed / len(test_results)


class MockLLMAgentFactory:
    """Mock factory for creating LLM agents"""
    
    @staticmethod
    def create_agent(config):
        """Create a mock LLM agent"""
        return MockLLMAgent(config)


class MockStateManager:
    """Mock state manager"""
    
    def __init__(self, state_path=None):
        self.state_path = state_path
    
    async def save_state(self, container_id, snapshot_id, working_dir="/workspace"):
        """Mock state saving"""
        return {"snapshot_id": snapshot_id, "success": True}
    
    async def restore_state(self, container_id, snapshot_id, working_dir="/workspace"):
        """Mock state restoration"""
        return True


class RealE2ETest:
    """Real end-to-end test that runs the complete framework"""
    
    def __init__(self):
        self.temp_dir = None
        self.original_config_path = None
        
    def setup_test_environment(self):
        """Set up test environment with real config"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Copy the real calculator API config
        self.original_config_path = Path(__file__).parent.parent / "examples/calculator_api/config.json"
        test_config_path = Path(self.temp_dir) / "config.json"
        
        # Load and modify the config for testing
        with open(self.original_config_path, 'r') as f:
            config_data = json.load(f)
        
        # Modify config for testing
        config_data["rollout_config"]["max_parallel_rollouts"] = 1
        config_data["rollout_config"]["trajectory_output_path"] = str(Path(self.temp_dir) / "trajectories")
        config_data["timeout_config"]["global_timeout"] = 120
        config_data["timeout_config"]["command_timeout"] = 30
        config_data["timeout_config"]["test_timeout"] = 20
        
        # Save modified config
        with open(test_config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return test_config_path
    
    def create_test_llm_config(self):
        """Create test LLM configuration"""
        return LLMConfig(
            model="anthropic/claude-4-sonnet",
            api_key="test-api-key",
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=2000,
            timeout=60
        )
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    async def test_full_trajectory_collection(self):
        """Test complete trajectory collection process"""
        print("🚀 Testing full trajectory collection process...")
        
        try:
            # Setup
            config_path = self.setup_test_environment()
            
            # Load configuration
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Create framework config with LLM config
            framework_config = FrameworkConfig(**config_data)
            framework_config.llm_config = self.create_test_llm_config()
            
            print(f"📋 Configuration loaded:")
            print(f"  - Environments: {len(framework_config.environments)}")
            print(f"  - Environment ID: {framework_config.environments[0].id}")
            print(f"  - LLM Model: {framework_config.llm_config.model}")
            print(f"  - Output Path: {framework_config.rollout_config.trajectory_output_path}")
            
            # Mock the LLM agent and environment to avoid API calls and Docker
            with patch('src.core.rollout_manager.LLMAgent', MockLLMAgent), \
                 patch('src.core.rollout_manager.Environment', MockEnvironment), \
                 patch('src.core.rollout_manager.UnitTestRunner', MockTestRunner), \
                 patch('src.core.rollout_manager.LLMAgentFactory', MockLLMAgentFactory), \
                 patch('src.core.rollout_manager.StateManager', MockStateManager):
                
                # Create rollout manager
                rollout_manager = RolloutManager(framework_config)
                
                print("🔧 Initializing rollout manager...")
                success = await rollout_manager.initialize()
                assert success, "Rollout manager should initialize successfully"
                
                print("🏃 Running rollouts...")
                try:
                    # Run rollouts (this will collect trajectories)
                    results = await rollout_manager.run_rollouts()
                    
                    print(f"📊 Rollouts completed:")
                    print(f"  - Results count: {len(results)}")
                    
                    # Verify results
                    assert len(results) > 0, "Should have at least one result"
                    
                    # Check trajectory files were created
                    trajectory_dir = Path(framework_config.rollout_config.trajectory_output_path)
                    assert trajectory_dir.exists(), "Trajectory directory should exist"
                    
                    trajectory_files = list(trajectory_dir.glob("*.json"))
                    print(f"  - Trajectory files created: {len(trajectory_files)}")
                    assert len(trajectory_files) > 0, "Should have at least one trajectory file"
                    
                    # Verify trajectory content
                    for trajectory_file in trajectory_files:
                        with open(trajectory_file, 'r') as f:
                            trajectory_data = json.load(f)
                        
                        print(f"📝 Trajectory {trajectory_file.name}:")
                        print(f"  - Environment: {trajectory_data['environment_id']}")
                        print(f"  - Steps: {len(trajectory_data.get('steps', []))}")
                        print(f"  - Test results: {len(trajectory_data.get('test_results', []))}")
                        
                        # Verify required fields
                        assert 'trajectory_id' in trajectory_data, "Should have trajectory_id"
                        assert 'environment_id' in trajectory_data, "Should have environment_id"
                        assert 'steps' in trajectory_data, "Should have steps"
                        assert trajectory_data['environment_id'] == 'calculator_api', "Should be calculator_api environment"
                        
                        # Verify steps have the right structure
                        if trajectory_data['steps']:
                            step = trajectory_data['steps'][0]
                            assert 'action' in step, "Step should have action"
                            assert 'result' in step, "Step should have result"
                            assert 'timestamp' in step, "Step should have timestamp"
                    
                    # Export results
                    results_file = Path(self.temp_dir) / "results.json"
                    await rollout_manager.export_results(str(results_file))
                    
                    assert results_file.exists(), "Results file should be created"
                    
                    with open(results_file, 'r') as f:
                        results_data = json.load(f)
                    
                    print(f"📄 Results file created:")
                    if isinstance(results_data, dict):
                        print(f"  - Episodes: {len(results_data.get('episodes', []))}")
                        print(f"  - Summary included: {'summary' in results_data}")
                        
                        # Verify results file structure
                        assert 'episodes' in results_data, "Results should have episodes"
                        assert 'summary' in results_data, "Results should have summary"
                    else:
                        print(f"  - Results: {len(results_data)}")
                        print(f"  - Results type: {type(results_data)}")
                        
                        # For list format, just check we have results
                        assert len(results_data) > 0, "Results should have data"
                    
                    print("✅ Full trajectory collection test passed!")
                    return True
                    
                finally:
                    # Always cleanup
                    await rollout_manager.cleanup()
                    
        except Exception as e:
            print(f"❌ Full trajectory collection test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup_test_environment()
    
    async def test_trajectory_content_validation(self):
        """Test that trajectories contain expected content for calculator API"""
        print("🔍 Testing trajectory content validation...")
        
        try:
            # Setup
            config_path = self.setup_test_environment()
            
            # Load configuration
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            framework_config = FrameworkConfig(**config_data)
            framework_config.llm_config = self.create_test_llm_config()
            
            # Mock the LLM agent and environment with specific expected actions
            with patch('src.core.rollout_manager.LLMAgent', MockLLMAgent), \
                 patch('src.core.rollout_manager.Environment', MockEnvironment), \
                 patch('src.core.rollout_manager.UnitTestRunner', MockTestRunner), \
                 patch('src.core.rollout_manager.LLMAgentFactory', MockLLMAgentFactory), \
                 patch('src.core.rollout_manager.StateManager', MockStateManager):
                
                rollout_manager = RolloutManager(framework_config)
                
                await rollout_manager.initialize()
                
                try:
                    # Run one rollout
                    results = await rollout_manager.run_rollouts()
                    
                    # Check trajectory content
                    trajectory_dir = Path(framework_config.rollout_config.trajectory_output_path)
                    trajectory_files = list(trajectory_dir.glob("*.json"))
                    
                    assert len(trajectory_files) > 0, "Should have trajectory files"
                    
                    # Load and validate the first trajectory
                    with open(trajectory_files[0], 'r') as f:
                        trajectory_data = json.load(f)
                    
                    steps = trajectory_data.get('steps', [])
                    print(f"📈 Trajectory analysis:")
                    print(f"  - Total steps: {len(steps)}")
                    
                    # Check if we have the expected action types
                    action_types = [step['action']['type'] for step in steps]
                    print(f"  - Action types: {action_types}")
                    
                    # Should have file write and command actions
                    assert 'file_write' in action_types, "Should have file_write action"
                    assert 'command' in action_types, "Should have command action"
                    
                    # Check for done action
                    if 'done' in action_types:
                        done_step = next(step for step in steps if step['action']['type'] == 'done')
                        print(f"  - Done message: {done_step['action']['content'][:50]}...")
                    
                    print("✅ Trajectory content validation test passed!")
                    return True
                    
                finally:
                    await rollout_manager.cleanup()
                    
        except Exception as e:
            print(f"❌ Trajectory content validation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup_test_environment()
    
    async def test_parallel_rollouts(self):
        """Test parallel rollout execution"""
        print("🔄 Testing parallel rollout execution...")
        
        try:
            # Setup with multiple parallel rollouts
            config_path = self.setup_test_environment()
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Set up for 2 parallel rollouts
            config_data["rollout_config"]["max_parallel_rollouts"] = 2
            
            framework_config = FrameworkConfig(**config_data)
            framework_config.llm_config = self.create_test_llm_config()
            
            with patch('src.core.rollout_manager.LLMAgent', MockLLMAgent), \
                 patch('src.core.rollout_manager.Environment', MockEnvironment), \
                 patch('src.core.rollout_manager.UnitTestRunner', MockTestRunner), \
                 patch('src.core.rollout_manager.LLMAgentFactory', MockLLMAgentFactory), \
                 patch('src.core.rollout_manager.StateManager', MockStateManager):
                
                rollout_manager = RolloutManager(framework_config)
                
                await rollout_manager.initialize()
                
                try:
                    # Run rollouts
                    results = await rollout_manager.run_rollouts()
                    
                    # Check that we got results
                    print(f"📊 Parallel rollouts completed:")
                    print(f"  - Results: {len(results)}")
                    
                    # Check trajectory files
                    trajectory_dir = Path(framework_config.rollout_config.trajectory_output_path)
                    trajectory_files = list(trajectory_dir.glob("*.json"))
                    print(f"  - Trajectory files: {len(trajectory_files)}")
                    
                    # Should have at least one trajectory
                    assert len(trajectory_files) > 0, "Should have trajectory files"
                    
                    print("✅ Parallel rollout test passed!")
                    return True
                    
                finally:
                    await rollout_manager.cleanup()
                    
        except Exception as e:
            print(f"❌ Parallel rollout test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup_test_environment()
    
    async def run_all_tests(self):
        """Run all real end-to-end tests"""
        print("🎯 Starting Real End-to-End Framework Tests\n")
        
        tests = [
            self.test_full_trajectory_collection,
            self.test_trajectory_content_validation,
            self.test_parallel_rollouts,
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            print(f"\n{'='*60}")
            try:
                result = await test()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Test {test.__name__} failed with exception: {e}")
                failed += 1
            print(f"{'='*60}")
        
        print(f"\n{'='*60}")
        print(f"🎯 Real End-to-End Test Results:")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
        print(f"{'='*60}")
        
        return failed == 0


async def main():
    """Run the real end-to-end tests"""
    test_suite = RealE2ETest()
    success = await test_suite.run_all_tests()
    
    if success:
        print("\n🎉 All real end-to-end tests passed!")
        print("The framework successfully collects trajectories end-to-end!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 