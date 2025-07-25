# Real Work by Ritser Labs
An extensible, modular framework for reinforcement learning with LLM agents that work in Docker environments to simulate realistic workflows.

Currently, it is implemented to simulate software development workflows.


![Screenshot of Real Work](/docs/screenshot.png)

## Features

- **Modular Architecture**: Plugin-based system for extensible functionality
- **Docker Environments**: Isolated execution environments for each rollout
- **LLM Integration**: Native OpenAI tool calling with OpenRouter support
- **Advanced Tool Calling**: Native function calling instead of text parsing
- **Parallel Execution**: Support for running multiple rollouts simultaneously
- **Trajectory Tracking**: Comprehensive logging and analysis of agent behavior
- **State Persistence**: File system snapshots for environment state management
- **Unit Test Evaluation**: Automated testing and scoring of agent outputs
- **Dynamic Rollout Management**: Plugins can modify rollout behavior based on performance

## Installation

### Prerequisites

- **Docker**: This framework requires Docker to be installed and running on your system. Download and install Docker from [https://www.docker.com/get-started](https://www.docker.com/get-started)
- **Python 3.8+**: Required for running the framework

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/ritser-labs/real-work
cd real-work
```

2. Install dependencies:
```bash
pip install uv
uv sync
```

3. Verify Docker is installed and running:
```bash
docker --version
docker run hello-world
```

4. Set up your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

Or set it directly in the configuration file.

## Quick Start

1. **Set up your configuration file** (see `examples/calculator_api/config.json`):
```json
{
  "environments": [
    {
      "id": "my_env",
      "docker_image": "python:3.9-slim",
      "init_commands": ["pip install --user flask pytest"],
      "unit_tests": ["python -m pytest tests/"],
      "prompt": "Your task description here...",
      "working_directory": "/tmp",
      "environment_variables": {
        "PYTHONPATH": "/tmp",
        "HOME": "/tmp",
        "PATH": "/tmp/.local/bin:/usr/local/bin:/usr/bin:/bin"
      },
      "copy_folders": [],
      "max_retries": 3
    }
  ],
  "rollout_config": {
    "max_parallel_rollouts": 4,
    "trajectory_output_path": "trajectories",
    "enable_plugins": true,
    "save_trajectory_interval": 10,
    "state_persistence_enabled": true
  },
  "episode_control_config": {
    "max_episodes": 1,
    "max_episodes_per_env": null,
    "stop_on_success": false,
    "min_success_rate": null,
    "safety_limit": 100
  },
  "timeout_config": {
    "global_timeout": 1800,
    "command_timeout": 300,
    "test_timeout": 120
  },
  "template_prompt": "You are an expert software engineer working in a Docker container environment. You have access to shell commands, can read and write files, and should implement the requested functionality step by step.",
  "plugins": []
}
```

2. **Run the framework**:
```bash
uv run main.py examples/calculator_api/config.json \
  --llm-api-key your-api-key-here \
  --llm-model anthropic/claude-sonnet-4 \
  --llm-base-url https://openrouter.ai/api/v1
```

**LLM Configuration Options**:
- `--llm-model`: LLM model to use (default: anthropic/claude-sonnet-4)
- `--llm-api-key`: API key for the LLM service (required)
- `--llm-base-url`: Base URL for the LLM API (default: https://openrouter.ai/api/v1)
- `--llm-temperature`: Temperature for sampling (default: 0.7)
- `--llm-max-tokens`: Maximum tokens for response (default: 4096)
- `--llm-timeout`: Timeout for API calls in seconds (default: 60)

3. **View results**:
Results are saved as JSON files in the specified output directory. The framework provides detailed trajectory information and performance statistics.

## Framework Architecture

### Core Components

- **`RolloutManager`**: Orchestrates the entire process, manages parallel execution
- **`Environment`**: Handles Docker containers and command execution
- **`LLMAgent`**: Manages LLM API calls and action parsing
- **`TrajectoryManager`**: Tracks and persists episode trajectories
- **`UnitTestRunner`**: Executes tests and collects results
- **`PluginManager`**: Manages plugin lifecycle and execution

### Plugin System

The framework supports several types of plugins:

1. **RolloutManagerPlugin**: Controls rollout execution strategy
2. **TrajectoryAnalyzerPlugin**: Analyzes trajectories for insights
3. **ActionFilterPlugin**: Filters or modifies actions before execution
4. **EnvironmentModifierPlugin**: Modifies environment configuration
5. **EpisodeEvaluatorPlugin**: Provides custom episode evaluation

### Built-in Plugins

- **PerformanceBasedRolloutPlugin**: Stops rollouts when performance threshold is met
- **TrajectoryAnalysisPlugin**: Provides detailed trajectory analysis
- **SafetyFilterPlugin**: Blocks potentially dangerous commands

## Configuration

**Required Fields**: The following fields are required in your configuration file:
- `environments` - List of environment configurations
- `rollout_config` - Rollout execution settings
- `episode_control_config` - Episode control settings
- `timeout_config` - Global timeout settings
- `template_prompt` - Base prompt template

### Environment Configuration

```json
{
  "id": "unique_environment_id",
  "docker_image": "python:3.9-slim",
  "init_commands": ["pip install requirements"],
  "unit_tests": ["python -m pytest tests/"],
  "prompt": "Detailed task description for the LLM",
  "working_directory": "/workspace",
  "environment_variables": {
    "PYTHONPATH": "/workspace"
  },
  "copy_folders": [],
  "max_retries": 3
}
```

### LLM Configuration

**Note**: LLM configuration is typically provided via command-line arguments rather than in the JSON file. The framework supports the following CLI options:

```bash
--llm-model anthropic/claude-sonnet-4
--llm-api-key your-api-key-here
--llm-base-url https://openrouter.ai/api/v1
--llm-temperature 0.7
--llm-max-tokens 4096
--llm-timeout 60
```

For advanced LLM configuration, you can also specify:

```json
{
  "model": "anthropic/claude-sonnet-4",
  "api_key": "your-api-key",
  "base_url": "https://openrouter.ai/api/v1",
  "temperature": 0.7,
  "max_tokens": 4096,
  "timeout": 60,
  "enable_caching": true,
  "cache_size": 100,
  "max_context_messages": 50,
  "max_output_length": 2000,
  "track_token_usage": true,
  "warn_high_usage": true
}
```

### Rollout Configuration

```json
{
  "max_parallel_rollouts": 4,
  "trajectory_output_path": "trajectories",
  "enable_plugins": true,
  "save_trajectory_interval": 10,
  "state_persistence_enabled": true
}
```

### Episode Control Configuration

```json
{
  "max_episodes": 1,
  "max_episodes_per_env": null,
  "stop_on_success": false,
  "min_success_rate": null,
  "safety_limit": 100
}
```

### Timeout Configuration

```json
{
  "global_timeout": 1800,
  "command_timeout": 300,
  "test_timeout": 120
}
```

## LLM Tool Calling

The framework uses OpenAI's native tool calling functionality. The LLM has access to the following tools:

### Available Tools

1. **execute_command** - Execute shell commands
   - `command`: The shell command to execute
   - `timeout`: Optional timeout in seconds
   - `working_directory`: Optional working directory

2. **write_file** - Write content to files
   - `filepath`: Path to the file to write
   - `content`: Content to write to the file

3. **read_file** - Read content from files
   - `filepath`: Path to the file to read

4. **mark_done** - Mark task as completed
   - `message`: Optional completion message

The LLM automatically calls these tools as needed, and the framework handles the execution and returns results.

## Example: Calculator API

The framework includes a complete example that demonstrates building a calculator API:

```bash
# Run the calculator API example
uv run python main.py examples/calculator_api/config.json --output calculator_results.json

# View results
cat calculator_results.json
cat calculator_results.stats.json
```

## Creating Custom Plugins

### Example Plugin

```python
from src.plugins.base import TrajectoryAnalyzerPlugin, PluginHookType

class CustomAnalyzerPlugin(TrajectoryAnalyzerPlugin):
    @property
    def name(self) -> str:
        return "custom_analyzer"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_hooks(self) -> List[PluginHookType]:
        return [PluginHookType.POST_EPISODE]
    
    async def analyze_trajectory(self, trajectory: Trajectory) -> Dict[str, Any]:
        # Your custom analysis logic here
        return {"custom_metric": 42}
    
    async def suggest_improvements(self, trajectory: Trajectory) -> List[str]:
        return ["Custom suggestion based on analysis"]
```

### Loading Custom Plugins

Add your plugin module to the configuration:

```json
{
  "plugins": ["my_custom_plugins.analyzer"]
}
```

## Output Format

### Trajectory JSON Structure

```json
{
  "trajectory_id": "uuid",
  "environment_id": "env_id",
  "start_time": "2024-01-01T00:00:00",
  "end_time": "2024-01-01T01:00:00",
  "steps": [
    {
      "step_id": "uuid",
      "timestamp": "2024-01-01T00:00:00",
      "action": {
        "type": "command",
        "content": "ls -la",
        "timeout": 300
      },
      "result": {
        "success": true,
        "output": "file listing...",
        "duration": 1.5
      }
    }
  ],
  "test_results": [
    {
      "command": "python -m pytest",
      "success": true,
      "output": "test output...",
      "duration": 5.2
    }
  ],
  "episode_result": {
    "success": true,
    "final_score": 0.95,
    "terminated_reason": "completed"
  }
}
```

## Performance Monitoring

The framework provides comprehensive performance monitoring:

- **Success rates** per environment
- **Average episode duration**
- **Test pass rates**
- **Action efficiency metrics**
- **Plugin performance statistics**

## Extending the Framework

### Adding New Action Types

1. Extend the `ActionType` enum in `src/core/config.py`
2. Add parsing logic in `src/agents/llm_agent.py`
3. Implement execution logic in `src/environments/environment.py`

### Adding New Environment Types

1. Extend the `Environment` class
2. Implement environment-specific initialization
3. Add custom test runners if needed

### Adding New Evaluation Metrics

1. Create an `EpisodeEvaluatorPlugin`
2. Implement custom scoring logic
3. Register the plugin in your configuration


### Configuration Validation

Validate your configuration without running:

```bash
uv run python main.py config.json --dry-run
```

**Note**: Some fields like `llm_model`, `max_episodes`, `stop_on_success`, `max_parallel_rollouts`, `global_timeout`, `output_path`, and `plugins_enabled` at the root level are deprecated and should be moved to their respective configuration sections (`llm_config`, `episode_control_config`, `rollout_config`, `timeout_config`).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

- [ ] Support for additional LLM providers
- [ ] Web-based dashboard for monitoring
- [ ] Integration with popular RL frameworks
- [ ] Support for multi-agent environments
- [ ] Advanced trajectory analysis tools
- [ ] Containerized deployment options

## Security disclaimer

The LLM can execute code in the Docker container, so be
wary of advanced prompt injection attacks if you are using
custom environments or models.