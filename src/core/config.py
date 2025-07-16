from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum


class TimeoutConfig(BaseModel):
    """Configuration for timeouts"""
    global_timeout: int = Field(default=3600, description="Global timeout for entire episode in seconds")
    command_timeout: int = Field(default=300, description="Default timeout for individual commands in seconds")
    test_timeout: int = Field(default=120, description="Timeout for running unit tests in seconds")


class EnvironmentConfig(BaseModel):
    """Configuration for a single environment"""
    id: str = Field(..., description="Unique identifier for the environment")
    docker_image: str = Field(..., description="Docker image to use for the environment")
    init_commands: List[str] = Field(default_factory=list, description="Commands to run when initializing the environment")
    unit_tests: List[str] = Field(..., description="Unit test commands to run for evaluation")
    prompt: str = Field(..., description="Environment-specific prompt describing the task")
    timeout_config: Optional[TimeoutConfig] = Field(default=None, description="Environment-specific timeout configuration")
    working_directory: str = Field(default="/workspace", description="Working directory inside the container")
    environment_variables: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    copy_folders: List[str] = Field(default_factory=list, description="List of local folders to copy into the container")
    max_retries: int = Field(default=3, description="Maximum retries for failed operations")


class LLMConfig(BaseModel):
    """Configuration for LLM agent"""
    model: str = Field(default="gpt-4", description="Model to use for LLM")
    api_key: str = Field(..., description="API key for OpenAI/OpenRouter")
    base_url: str = Field(default="https://api.openai.com/v1", description="Base URL for API")
    temperature: float = Field(default=0.7, description="Temperature for sampling")
    max_tokens: int = Field(default=4096, description="Maximum tokens in response")
    timeout: int = Field(default=60, description="Timeout for API calls in seconds")
    
    # Token caching and context management
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_size: int = Field(default=100, description="Maximum number of cached responses")
    max_context_messages: int = Field(default=50, description="Maximum messages in context before truncation")
    max_output_length: int = Field(default=2000, description="Maximum length of command output to include in context")
    
    # Token usage tracking
    track_token_usage: bool = Field(default=True, description="Enable token usage tracking")
    max_cost_per_episode: Optional[float] = Field(default=None, description="Maximum cost per episode in USD")
    warn_high_usage: bool = Field(default=True, description="Warn when token usage is high")


class RolloutConfig(BaseModel):
    """Configuration for rollout execution"""
    max_parallel_rollouts: int = Field(default=4, description="Maximum number of parallel rollouts")
    save_trajectory_interval: int = Field(default=10, description="Save trajectory every N steps")
    enable_plugins: bool = Field(default=True, description="Enable plugin system")
    trajectory_output_path: str = Field(default="trajectories", description="Path to save trajectories")
    state_persistence_enabled: bool = Field(default=True, description="Enable state persistence")


class EpisodeControlConfig(BaseModel):
    """Configuration for episode control logic"""
    max_episodes: int = Field(default=1, description="Maximum number of episodes to run per environment")
    max_episodes_per_env: Optional[int] = Field(default=None, description="Override max episodes for specific environments")
    stop_on_success: bool = Field(default=False, description="Stop running episodes when one succeeds")
    min_success_rate: Optional[float] = Field(default=None, description="Minimum success rate to maintain (0.0-1.0)")
    safety_limit: int = Field(default=100, description="Absolute safety limit for episodes")


class FrameworkConfig(BaseModel):
    """Main configuration for the framework"""
    environments: List[EnvironmentConfig] = Field(..., description="List of environments to run")
    llm_config: Optional[LLMConfig] = Field(default=None, description="LLM configuration (can be provided via CLI)")
    rollout_config: RolloutConfig = Field(..., description="Rollout configuration")
    timeout_config: TimeoutConfig = Field(default_factory=TimeoutConfig, description="Global timeout configuration")
    template_prompt: str = Field(..., description="Template prompt that gets combined with environment-specific prompts")
    plugins: List[str] = Field(default_factory=list, description="List of plugin modules to load")
    episode_control_config: EpisodeControlConfig = Field(default_factory=EpisodeControlConfig, description="Episode control configuration")


class ActionType(str, Enum):
    """Types of actions the LLM can take"""
    COMMAND = "command"
    FILE_WRITE = "file_write"
    FILE_READ = "file_read"
    DONE = "done"


class Action(BaseModel):
    """Represents an action to be taken by the agent"""
    type: ActionType
    content: str
    timeout: Optional[int] = None
    working_directory: Optional[str] = None
    background: bool = False


class ActionResult(BaseModel):
    """Represents the result of an action"""
    success: bool = Field(..., description="Whether the action was successful")
    output: str = Field(default="", description="Output from the action")
    error: str = Field(default="", description="Error message if action failed")
    exit_code: Optional[int] = Field(default=None, description="Exit code for commands")
    duration: float = Field(..., description="Duration of action execution in seconds")
    timestamp: str = Field(..., description="Timestamp when action was executed")


class TestResult(BaseModel):
    """Result of unit test execution"""
    command: str = Field(..., description="Test command that was run")
    success: bool = Field(..., description="Whether tests passed")
    output: str = Field(..., description="Test output")
    error: str = Field(default="", description="Error output")
    exit_code: int = Field(..., description="Exit code from test execution")
    duration: float = Field(..., description="Duration of test execution")
    timestamp: str = Field(..., description="Timestamp when test was run")


class EpisodeResult(BaseModel):
    """Result of a complete episode"""
    environment_id: str = Field(..., description="Environment ID")
    success: bool = Field(..., description="Whether episode was successful")
    total_actions: int = Field(..., description="Total number of actions taken")
    duration: float = Field(..., description="Total duration of episode")
    test_results: List[TestResult] = Field(..., description="Unit test results")
    terminated_reason: str = Field(..., description="Reason for termination (done, timeout, error)")
    final_score: float = Field(default=0.0, description="Final score based on test results")
    
    # Token usage statistics
    token_usage: Optional[Dict[str, Any]] = Field(default=None, description="Token usage statistics for the episode") 