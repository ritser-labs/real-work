# Real Work by Ritser Labs

An extensible, modular framework for reinforcement learning with LLM agents that work in Docker environments to simulate realistic workflows.

Currently, it is implemented to simulate software development workflows.

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
- **Claude 3.5 Sonnet**: Uses state-of-the-art Claude 3.5 Sonnet model by default

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-rl-framework
```

2. Install dependencies:
```bash
pip install uv
uv install
```

3. Ensure Docker is installed and running on your system.

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
      "init_commands": ["pip install flask pytest"],
      "unit_tests": ["python -m pytest tests/"],
      "prompt": "Your task description here...",
      "working_directory": "/workspace"
    }
  ],
  "rollout_config": {
    "max_parallel_rollouts": 4,
    "trajectory_output_path": "trajectories"
  }
}
```

2. **Run the framework**:
```bash
python main.py examples/calculator_api/config.json \
  --llm-api-key your-api-key-here \
  --llm-model anthropic/claude-4-sonnet \
  --llm-base-url https://openrouter.ai/api/v1
```

**LLM Configuration Options**:
- `--llm-model`: LLM model to use (default: anthropic/claude-4-sonnet)
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
  "timeout_config": {
    "global_timeout": 1800,
    "command_timeout": 300
  }
}
```

### LLM Configuration

```json
{
  "model": "gpt-4",
  "api_key": "your-api-key",
  "base_url": "https://api.openai.com/v1",
  "temperature": 0.7,
  "max_tokens": 4096
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
python main.py examples/calculator_api/config.json --output calculator_results.json

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

## Security Considerations

- Environments run with non-root user (`1000:1000`)
- Memory and CPU limits on Docker containers
- SafetyFilterPlugin blocks dangerous commands
- Network isolation for containers
- Configurable command timeouts

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

## Troubleshooting

### Common Issues

1. **Docker not found**: Ensure Docker is installed and running
2. **API key issues**: Check your OpenAI API key configuration
3. **Permission errors**: Ensure proper file permissions for workspace
4. **Timeout errors**: Adjust timeout configuration for your use case

### Debug Mode

Run with debug logging for detailed information:

```bash
python main.py config.json --log-level DEBUG
```

### Configuration Validation

Validate your configuration without running:

```bash
python main.py config.json --dry-run
```

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

## Support

For questions and support, please open an issue on the GitHub repository.
