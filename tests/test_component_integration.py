#!/usr/bin/env python3
"""
Component Integration Test for the LLM RL Framework

This test validates individual framework components and their integration including:
- Configuration loading without llm_config
- Command-line LLM configuration
- Component initialization and setup
- Trajectory manager functionality
- Mock-based component interaction testing

Note: For real end-to-end testing, see test_real_e2e.py
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

from src.core.config import FrameworkConfig, LLMConfig, EnvironmentConfig, RolloutConfig
from src.core.rollout_manager import RolloutManager
from src.core.trajectory import TrajectoryManager
from src.environments.environment import Environment
from src.agents.llm_agent import LLMAgent


class TestComponentIntegration:
    """Component integration test suite for the framework"""
    
    def __init__(self):
        self.temp_dir = None
        self.test_config = None
        self.framework_config = None
        
    def setup_test_environment(self):
        """Set up temporary test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test config without llm_config
        self.test_config = {
            "environments": [
                {
                    "id": "test_simple",
                    "docker_image": "python:3.9-slim",
                    "init_commands": [
                        "pip install --user requests",
                        "mkdir -p /tmp"
                    ],
                    "unit_tests": [
                        "python -c \"print('Hello, World!')\"",
                        "python -c \"import requests; print('requests imported successfully')\""
                    ],
                    "prompt": "Create a simple Python script that prints 'Hello, World!' and demonstrates that requests library is available.",
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
                "trajectory_output_path": str(Path(self.temp_dir) / "trajectories"),
                "enable_plugins": False
            },
            "timeout_config": {
                "global_timeout": 300,
                "command_timeout": 60,
                "test_timeout": 30
            },
            "template_prompt": "You are a helpful assistant. Complete the given task step by step. When finished, use <done> to indicate completion.",
            "plugins": []
        }
        
        # Save test config file
        config_path = Path(self.temp_dir) / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)
        
        return config_path
    
    def create_llm_config(self):
        """Create LLM configuration"""
        return LLMConfig(
            model="anthropic/claude-4-sonnet",
            api_key="test-api-key",
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=1000,
            timeout=60
        )
    
    def cleanup_test_environment(self):
        """Clean up temporary test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    async def test_config_loading_without_llm_config(self):
        """Test that configuration loads properly without llm_config"""
        print("🧪 Testing configuration loading without llm_config...")
        
        try:
            config_path = self.setup_test_environment()
            
            # Load config from file
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Create framework config
            framework_config = FrameworkConfig(**config_data)
            
            # Verify that llm_config is None
            assert framework_config.llm_config is None, "llm_config should be None when not provided"
            
            # Verify other config is loaded correctly
            assert len(framework_config.environments) == 1, "Should have 1 environment"
            assert framework_config.environments[0].id == "test_simple", "Environment ID should match"
            assert framework_config.rollout_config.max_parallel_rollouts == 1, "Rollout config should match"
            
            print("✅ Configuration loading test passed")
            return True
            
        except Exception as e:
            print(f"❌ Configuration loading test failed: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    async def test_llm_config_injection(self):
        """Test that LLM config can be injected after loading"""
        print("🧪 Testing LLM config injection...")
        
        try:
            config_path = self.setup_test_environment()
            
            # Load config from file
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Create framework config
            framework_config = FrameworkConfig(**config_data)
            
            # Inject LLM config
            llm_config = self.create_llm_config()
            framework_config.llm_config = llm_config
            
            # Verify LLM config is set
            assert framework_config.llm_config is not None, "llm_config should not be None after injection"
            assert framework_config.llm_config.model == "anthropic/claude-4-sonnet", "Model should match"
            assert framework_config.llm_config.api_key == "test-api-key", "API key should match"
            
            print("✅ LLM config injection test passed")
            return True
            
        except Exception as e:
            print(f"❌ LLM config injection test failed: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    async def test_rollout_manager_initialization(self):
        """Test that rollout manager initializes with injected LLM config"""
        print("🧪 Testing rollout manager initialization...")
        
        try:
            config_path = self.setup_test_environment()
            
            # Load and prepare config
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            framework_config = FrameworkConfig(**config_data)
            framework_config.llm_config = self.create_llm_config()
            
            # Create rollout manager
            rollout_manager = RolloutManager(framework_config)
            
            # Verify initialization
            assert rollout_manager.config == framework_config, "Config should be set"
            assert rollout_manager.config.llm_config is not None, "LLM config should be available"
            
            print("✅ Rollout manager initialization test passed")
            return True
            
        except Exception as e:
            print(f"❌ Rollout manager initialization test failed: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    @patch('src.agents.llm_agent.LLMAgent')
    @patch('src.environments.environment.Environment')
    async def test_mock_trajectory_generation(self, mock_env_class, mock_agent_class):
        """Test trajectory generation with mocked LLM and environment"""
        print("🧪 Testing trajectory generation with mocks...")
        
        try:
            config_path = self.setup_test_environment()
            
            # Load and prepare config
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            framework_config = FrameworkConfig(**config_data)
            framework_config.llm_config = self.create_llm_config()
            
            # Mock environment
            mock_env = Mock()
            mock_env.initialize = AsyncMock(return_value=True)
            mock_env.cleanup = AsyncMock()
            mock_env.config = framework_config.environments[0]
            mock_env_class.return_value = mock_env
            
            # Mock LLM agent
            mock_agent = Mock()
            mock_agent.initialize_conversation = AsyncMock()
            mock_agent.get_next_action = AsyncMock(return_value=[])
            mock_agent_class.return_value = mock_agent
            
            # Create trajectory manager
            trajectory_manager = TrajectoryManager(str(Path(self.temp_dir) / "trajectories"))
            
            # Test trajectory creation
            trajectory_id = trajectory_manager.create_trajectory(
                environment_id="test_simple",
                metadata={"agent_config": framework_config.llm_config.model_dump()}
            )
            
            assert trajectory_id is not None, "Trajectory ID should not be None"
            assert trajectory_id in trajectory_manager.trajectories, "Trajectory should be stored"
            
            print("✅ Trajectory generation test passed")
            return True
            
        except Exception as e:
            print(f"❌ Trajectory generation test failed: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    async def test_trajectory_file_creation(self):
        """Test that trajectory files are created properly"""
        print("🧪 Testing trajectory file creation...")
        
        try:
            config_path = self.setup_test_environment()
            
            # Create trajectory manager
            trajectory_dir = Path(self.temp_dir) / "trajectories"
            trajectory_manager = TrajectoryManager(str(trajectory_dir))
            
            # Create a test trajectory
            trajectory_id = trajectory_manager.create_trajectory(
                environment_id="test_simple",
                metadata={"agent_config": {"model": "anthropic/claude-4-sonnet"}}
            )
            
            # Save trajectory
            await trajectory_manager.save_all_trajectories()
            
            # Verify file exists
            trajectory_file = trajectory_dir / f"{trajectory_id}.json"
            assert trajectory_file.exists(), "Trajectory file should exist"
            
            # Verify file content
            with open(trajectory_file, 'r') as f:
                trajectory_data = json.load(f)
            
            assert trajectory_data["environment_id"] == "test_simple", "Environment ID should match"
            assert trajectory_data["metadata"]["agent_config"]["model"] == "anthropic/claude-4-sonnet", "Model should match"
            
            print("✅ Trajectory file creation test passed")
            return True
            
        except Exception as e:
            print(f"❌ Trajectory file creation test failed: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    async def test_complete_workflow(self):
        """Test the complete workflow with all components"""
        print("🧪 Testing complete workflow...")
        
        try:
            config_path = self.setup_test_environment()
            
            # Load and prepare config
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            framework_config = FrameworkConfig(**config_data)
            framework_config.llm_config = self.create_llm_config()
            
            # Create trajectory manager
            trajectory_dir = Path(self.temp_dir) / "trajectories"
            trajectory_manager = TrajectoryManager(str(trajectory_dir))
            
            # Test full workflow components
            trajectory_id = trajectory_manager.create_trajectory(
                environment_id="test_simple",
                metadata={"agent_config": framework_config.llm_config.model_dump()}
            )
            
            # Simulate adding some test results
            from src.core.config import TestResult
            test_results = [
                TestResult(
                    command="python -c \"print('Hello, World!')\"",
                    success=True,
                    output="Hello, World!\n",
                    error="",
                    exit_code=0,
                    duration=0.1,
                    timestamp="2024-01-01T00:00:00"
                )
            ]
            
            await trajectory_manager.add_test_results(trajectory_id, test_results)
            
            # Save trajectory
            await trajectory_manager.save_all_trajectories()
            
            # Verify complete trajectory
            trajectory_file = trajectory_dir / f"{trajectory_id}.json"
            assert trajectory_file.exists(), "Trajectory file should exist"
            
            with open(trajectory_file, 'r') as f:
                trajectory_data = json.load(f)
            
            assert len(trajectory_data["test_results"]) == 1, "Should have 1 test result"
            assert trajectory_data["test_results"][0]["success"] == True, "Test should be successful"
            
            print("✅ Complete workflow test passed")
            return True
            
        except Exception as e:
            print(f"❌ Complete workflow test failed: {e}")
            return False
        finally:
            self.cleanup_test_environment()
    
    async def run_all_tests(self):
        """Run all end-to-end tests"""
        print("🚀 Starting Component Integration Tests\n")
        
        tests = [
            self.test_config_loading_without_llm_config,
            self.test_llm_config_injection,
            self.test_rollout_manager_initialization,
            self.test_mock_trajectory_generation,
            self.test_trajectory_file_creation,
            self.test_complete_workflow
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                result = await test()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Test {test.__name__} failed with exception: {e}")
                failed += 1
            print()  # Add spacing between tests
        
        print("=" * 50)
        print(f"Component Integration Test Results:")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
        
        return failed == 0


async def main():
    """Run the component integration tests"""
    test_suite = TestComponentIntegration()
    success = await test_suite.run_all_tests()
    
    if success:
        print("\n🎉 All component integration tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 