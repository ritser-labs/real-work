from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from ..core.config import EnvironmentConfig, Action, ActionResult, TestResult, EpisodeResult
from ..core.trajectory import Trajectory


class PluginType(str, Enum):
    """Types of plugins supported by the framework"""
    ROLLOUT_MANAGER = "rollout_manager"
    TRAJECTORY_ANALYZER = "trajectory_analyzer"
    ENVIRONMENT_MODIFIER = "environment_modifier"
    ACTION_FILTER = "action_filter"
    EPISODE_EVALUATOR = "episode_evaluator"


class PluginHookType(str, Enum):
    """Hook points where plugins can be executed"""
    PRE_ROLLOUT = "pre_rollout"
    POST_ROLLOUT = "post_rollout"
    PRE_ACTION = "pre_action"
    POST_ACTION = "post_action"
    PRE_EPISODE = "pre_episode"
    POST_EPISODE = "post_episode"
    PRE_TEST = "pre_test"
    POST_TEST = "post_test"


class PluginContext:
    """Context information passed to plugins"""
    
    def __init__(self, **kwargs):
        self.data = kwargs
    
    def get(self, key: str, default=None):
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any):
        self.data[key] = value
    
    def update(self, data: Dict[str, Any]):
        self.data.update(data)


class BasePlugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = True
        self.priority = 0  # Higher priority plugins execute first
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass
    
    @property
    @abstractmethod
    def plugin_type(self) -> PluginType:
        """Type of plugin"""
        pass
    
    @property
    @abstractmethod
    def supported_hooks(self) -> List[PluginHookType]:
        """List of hooks this plugin supports"""
        pass
    
    @abstractmethod
    async def execute(self, hook: PluginHookType, context: PluginContext) -> PluginContext:
        """Execute plugin logic for a specific hook"""
        pass
    
    async def initialize(self) -> bool:
        """Initialize the plugin"""
        return True
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass
    
    def is_enabled(self) -> bool:
        """Check if plugin is enabled"""
        return self.enabled
    
    def enable(self):
        """Enable the plugin"""
        self.enabled = True
    
    def disable(self):
        """Disable the plugin"""
        self.enabled = False


class RolloutManagerPlugin(BasePlugin):
    """Base class for rollout management plugins"""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.ROLLOUT_MANAGER
    
    @abstractmethod
    async def should_continue_rollout(self, context: PluginContext) -> bool:
        """Determine if rollout should continue"""
        pass
    
    @abstractmethod
    async def modify_rollout_strategy(self, context: PluginContext) -> Dict[str, Any]:
        """Modify rollout strategy based on current performance"""
        pass


class TrajectoryAnalyzerPlugin(BasePlugin):
    """Base class for trajectory analysis plugins"""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.TRAJECTORY_ANALYZER
    
    @abstractmethod
    async def analyze_trajectory(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Analyze a trajectory and return insights"""
        pass
    
    @abstractmethod
    async def suggest_improvements(self, trajectory: Trajectory) -> List[str]:
        """Suggest improvements based on trajectory analysis"""
        pass


class EnvironmentModifierPlugin(BasePlugin):
    """Base class for environment modification plugins"""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.ENVIRONMENT_MODIFIER
    
    @abstractmethod
    async def modify_environment(self, env_config: EnvironmentConfig, context: PluginContext) -> EnvironmentConfig:
        """Modify environment configuration"""
        pass


class ActionFilterPlugin(BasePlugin):
    """Base class for action filtering plugins"""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.ACTION_FILTER
    
    @abstractmethod
    async def filter_actions(self, actions: List[Action], context: PluginContext) -> List[Action]:
        """Filter or modify actions before execution"""
        pass


class EpisodeEvaluatorPlugin(BasePlugin):
    """Base class for episode evaluation plugins"""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.EPISODE_EVALUATOR
    
    @abstractmethod
    async def evaluate_episode(self, episode_result: EpisodeResult, trajectory: Trajectory) -> Dict[str, Any]:
        """Evaluate an episode and return scoring/analysis"""
        pass


class PluginResult:
    """Result of plugin execution"""
    
    def __init__(self, success: bool = True, data: Dict[str, Any] = None, error: str = None):
        self.success = success
        self.data = data or {}
        self.error = error
    
    def __bool__(self):
        return self.success 