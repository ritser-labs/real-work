#!/usr/bin/env python3
"""
Test script to validate the LLM RL Framework installation and configuration.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
try:
    from src.core.config import FrameworkConfig, EnvironmentConfig, LLMConfig, RolloutConfig
    from src.core.trajectory import TrajectoryManager
    from src.core.test_runner import UnitTestRunner
    from src.core.rollout_manager import RolloutManager
    from src.environments.environment import Environment, StateManager
    from src.agents.llm_agent import LLMAgent
    from src.plugins.manager import PluginManager
    from src.plugins.builtin import PerformanceBasedRolloutPlugin, TrajectoryAnalysisPlugin
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test configuration loading
def test_config_loading():
    """Test configuration loading and validation"""
    print("\n=== Testing Configuration Loading ===")
    
    # Test valid configuration
    config_data = {
        "environments": [
            {
                "id": "test_env",
                "docker_image": "python:3.9-slim",
                "init_commands": ["pip install flask"],
                "unit_tests": ["python -c 'print(\"test\")'"],
                "prompt": "Test prompt",
                "working_directory": "/workspace"
            }
        ],
        "llm_config": {
            "model": "anthropic/claude-4-sonnet",
            "api_key": "test-key",
            "base_url": "https://openrouter.ai/api/v1"
        },
        "rollout_config": {
            "max_parallel_rollouts": 2,
            "trajectory_output_path": "test_trajectories"
        },
        "timeout_config": {
            "global_timeout": 300,
            "command_timeout": 60
        },
        "template_prompt": "Test template prompt"
    }
    
    try:
        config = FrameworkConfig(**config_data)
        print("✓ Configuration parsing successful")
        print(f"  - Environments: {len(config.environments)}")
        print(f"  - LLM Model: {config.llm_config.model}")
        print(f"  - Max Parallel Rollouts: {config.rollout_config.max_parallel_rollouts}")
        return True
    except Exception as e:
        print(f"✗ Configuration parsing failed: {e}")
        return False

# Test trajectory manager
def test_trajectory_manager():
    """Test trajectory manager functionality"""
    print("\n=== Testing Trajectory Manager ===")
    
    try:
        from src.core.config import Action, ActionResult, ActionType
        from datetime import datetime
        
        # Create trajectory manager
        tm = TrajectoryManager("test_trajectories", 5)
        
        # Create a test trajectory
        trajectory_id = tm.create_trajectory("test_env", {"test": "metadata"})
        print(f"✓ Created trajectory: {trajectory_id}")
        
        # Add a test step
        action = Action(
            type=ActionType.COMMAND,
            content="echo 'hello world'",
            timeout=30
        )
        
        result = ActionResult(
            success=True,
            output="hello world",
            duration=0.5,
            timestamp=datetime.now().isoformat()
        )
        
        asyncio.run(tm.add_step(trajectory_id, action, result))
        print("✓ Added trajectory step")
        
        # Get trajectory summary
        summary = tm.get_trajectory_summary(trajectory_id)
        print(f"✓ Trajectory summary: {summary['total_steps']} steps")
        
        return True
    except Exception as e:
        print(f"✗ Trajectory manager test failed: {e}")
        return False

# Test plugin system
def test_plugin_system():
    """Test plugin system functionality"""
    print("\n=== Testing Plugin System ===")
    
    try:
        # Create plugin manager
        pm = PluginManager()
        
        # Load a built-in plugin
        asyncio.run(pm.load_plugin(PerformanceBasedRolloutPlugin))
        print("✓ Loaded PerformanceBasedRolloutPlugin")
        
        # List plugins
        plugins = pm.list_plugins()
        print(f"✓ Plugin count: {len(plugins)}")
        
        # Get plugin statistics
        stats = pm.get_plugin_statistics()
        print(f"✓ Plugin statistics: {stats['total_plugins']} total, {stats['enabled_plugins']} enabled")
        
        return True
    except Exception as e:
        print(f"✗ Plugin system test failed: {e}")
        return False

# Test LLM agent tools
def test_llm_agent_tools():
    """Test LLM agent tool functionality"""
    print("\n=== Testing LLM Agent Tools ===")
    
    try:
        from src.core.config import LLMConfig
        
        # Create a test config
        config = LLMConfig(
            model="anthropic/claude-4-sonnet",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1"
        )
        
        agent = LLMAgent(config)
        
        # Test tool creation
        print(f"✓ Created LLM agent with {len(agent.tools)} tools")
        
        # Test tool mapping
        tool_names = list(agent.tool_mapping.keys())
        print(f"✓ Tool mapping: {tool_names}")
        
        # Test action creation
        command_action = agent._create_command_action(command="ls -la", timeout=30)
        print(f"✓ Created command action: {command_action.type}")
        
        file_write_action = agent._create_file_write_action(filepath="/test.py", content="print('hello')")
        print(f"✓ Created file write action: {file_write_action.type}")
        
        return True
    except Exception as e:
        print(f"✗ LLM agent tools test failed: {e}")
        return False

# Test configuration validation
def test_example_config():
    """Test the example configuration file"""
    print("\n=== Testing Example Configuration ===")
    
    config_path = Path(__file__).parent.parent / "examples/calculator_api/config.json"
    if not config_path.exists():
        print(f"✗ Example configuration not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Add test LLM config since it's no longer in the file
        config_data["llm_config"] = {
            "model": "anthropic/claude-4-sonnet",
            "api_key": "test-key",
            "base_url": "https://openrouter.ai/api/v1"
        }
        
        config = FrameworkConfig(**config_data)
        print("✓ Example configuration is valid")
        
        # Check environment details
        env = config.environments[0]
        print(f"  - Environment ID: {env.id}")
        print(f"  - Docker Image: {env.docker_image}")
        print(f"  - Unit Tests: {len(env.unit_tests)}")
        print(f"  - Init Commands: {len(env.init_commands)}")
        
        return True
    except Exception as e:
        print(f"✗ Example configuration test failed: {e}")
        return False

# Main test function
def main():
    """Run all tests"""
    print("LLM RL Framework - Installation Test")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_trajectory_manager,
        test_plugin_system,
        test_llm_agent_tools,
        test_example_config
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Framework is ready to use.")
        print("\nNext steps:")
        print("1. Set your OpenAI API key in examples/calculator_api/config.json")
        print("2. Run: python main.py examples/calculator_api/config.json --dry-run")
        print("3. Run: python main.py examples/calculator_api/config.json")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 