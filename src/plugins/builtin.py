import asyncio
from typing import Dict, Any, List
from statistics import mean

from .base import (
    RolloutManagerPlugin, TrajectoryAnalyzerPlugin, ActionFilterPlugin,
    PluginHookType, PluginContext, PluginType
)
from ..core.trajectory import Trajectory
from ..core.config import Action, ActionType, EpisodeResult


class PerformanceBasedRolloutPlugin(RolloutManagerPlugin):
    """Plugin that manages rollouts based on performance metrics"""
    
    @property
    def name(self) -> str:
        return "performance_rollout_manager"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_hooks(self) -> List[PluginHookType]:
        return [PluginHookType.POST_EPISODE, PluginHookType.POST_ROLLOUT]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.success_threshold = config.get("success_threshold", 0.8)
        self.min_episodes_per_env = config.get("min_episodes_per_env", 3)
        self.max_episodes_per_env = config.get("max_episodes_per_env", 10)
        self.performance_history = {}
    
    async def execute(self, hook: PluginHookType, context: PluginContext) -> PluginContext:
        """Execute plugin logic"""
        if hook == PluginHookType.POST_EPISODE:
            await self._record_episode_performance(context)
        elif hook == PluginHookType.POST_ROLLOUT:
            await self._evaluate_rollout_performance(context)
        
        return context
    
    async def _record_episode_performance(self, context: PluginContext) -> None:
        """Record episode performance"""
        episode_result = context.get("episode_result")
        environment_id = context.get("environment_id")
        
        if not episode_result or not environment_id:
            return
        
        if environment_id not in self.performance_history:
            self.performance_history[environment_id] = []
        
        # Calculate performance score
        score = self._calculate_performance_score(episode_result)
        self.performance_history[environment_id].append(score)
        
        context.set("performance_score", score)
    
    async def _evaluate_rollout_performance(self, context: PluginContext) -> None:
        """Evaluate overall rollout performance"""
        overall_performance = {}
        
        for env_id, scores in self.performance_history.items():
            if scores:
                overall_performance[env_id] = {
                    "mean_score": mean(scores),
                    "max_score": max(scores),
                    "min_score": min(scores),
                    "episode_count": len(scores),
                    "success_rate": sum(1 for s in scores if s >= self.success_threshold) / len(scores)
                }
        
        context.set("overall_performance", overall_performance)
    
    def _calculate_performance_score(self, episode_result: EpisodeResult) -> float:
        """Calculate a performance score for an episode"""
        if not episode_result.test_results:
            return 0.0
        
        # Calculate test success rate
        passed_tests = sum(1 for test in episode_result.test_results if test.success)
        total_tests = len(episode_result.test_results)
        test_success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Factor in completion status
        completion_bonus = 0.1 if episode_result.success else 0.0
        
        # Factor in efficiency (inverse of action count)
        efficiency_bonus = max(0.0, 0.2 - (episode_result.total_actions / 100))
        
        return min(1.0, test_success_rate + completion_bonus + efficiency_bonus)
    
    async def should_continue_rollout(self, context: PluginContext) -> bool:
        """Determine if rollout should continue"""
        environment_id = context.get("environment_id")
        
        if environment_id not in self.performance_history:
            return True
        
        scores = self.performance_history[environment_id]
        
        # Continue if we haven't reached minimum episodes
        if len(scores) < self.min_episodes_per_env:
            return True
        
        # Stop if we've reached maximum episodes
        if len(scores) >= self.max_episodes_per_env:
            return False
        
        # Stop if recent performance is consistently good
        if len(scores) >= 3:
            recent_scores = scores[-3:]
            if all(score >= self.success_threshold for score in recent_scores):
                return False
        
        return True
    
    async def modify_rollout_strategy(self, context: PluginContext) -> Dict[str, Any]:
        """Modify rollout strategy based on performance"""
        environment_id = context.get("environment_id")
        
        if environment_id not in self.performance_history:
            return {}
        
        scores = self.performance_history[environment_id]
        
        if not scores:
            return {}
        
        recent_performance = mean(scores[-3:]) if len(scores) >= 3 else mean(scores)
        
        modifications = {}
        
        # Adjust timeout based on performance
        if recent_performance < 0.3:
            modifications["timeout_multiplier"] = 1.5  # Give more time for poor performance
        elif recent_performance > 0.8:
            modifications["timeout_multiplier"] = 0.8  # Reduce time for good performance
        
        # Adjust retry count
        if recent_performance < 0.5:
            modifications["max_retries"] = 5
        else:
            modifications["max_retries"] = 3
        
        return modifications


class TrajectoryAnalysisPlugin(TrajectoryAnalyzerPlugin):
    """Plugin that analyzes trajectories to identify patterns and issues"""
    
    @property
    def name(self) -> str:
        return "trajectory_analyzer"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_hooks(self) -> List[PluginHookType]:
        return [PluginHookType.POST_EPISODE]
    
    async def execute(self, hook: PluginHookType, context: PluginContext) -> PluginContext:
        """Execute plugin logic"""
        if hook == PluginHookType.POST_EPISODE:
            trajectory = context.get("trajectory")
            if trajectory:
                analysis = await self.analyze_trajectory(trajectory)
                context.set("trajectory_analysis", analysis)
        
        return context
    
    async def analyze_trajectory(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Analyze a trajectory and return insights"""
        analysis = {
            "total_steps": len(trajectory.steps),
            "action_distribution": self._analyze_action_distribution(trajectory),
            "error_patterns": self._analyze_error_patterns(trajectory),
            "performance_metrics": self._analyze_performance_metrics(trajectory),
            "time_analysis": self._analyze_time_patterns(trajectory),
            "success_factors": self._analyze_success_factors(trajectory)
        }
        
        return analysis
    
    def _analyze_action_distribution(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Analyze distribution of action types"""
        action_counts = {}
        
        for step in trajectory.steps:
            action_type = step.action.type
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        total_actions = len(trajectory.steps)
        
        return {
            "counts": action_counts,
            "percentages": {
                action_type: (count / total_actions) * 100
                for action_type, count in action_counts.items()
            } if total_actions > 0 else {}
        }
    
    def _analyze_error_patterns(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Analyze error patterns in the trajectory"""
        errors = []
        error_types = {}
        
        for step in trajectory.steps:
            if not step.result.success:
                errors.append({
                    "step_id": step.step_id,
                    "action_type": step.action.type,
                    "error": step.result.error
                })
                
                # Categorize errors
                error_category = self._categorize_error(step.result.error)
                error_types[error_category] = error_types.get(error_category, 0) + 1
        
        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(trajectory.steps) if trajectory.steps else 0,
            "error_types": error_types,
            "errors": errors[-5:]  # Last 5 errors
        }
    
    def _analyze_performance_metrics(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Analyze performance metrics"""
        if not trajectory.steps:
            return {}
        
        durations = [step.result.duration for step in trajectory.steps]
        
        return {
            "avg_action_duration": mean(durations),
            "max_action_duration": max(durations),
            "min_action_duration": min(durations),
            "total_duration": sum(durations),
            "success_rate": sum(1 for step in trajectory.steps if step.result.success) / len(trajectory.steps)
        }
    
    def _analyze_time_patterns(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Analyze time patterns in the trajectory"""
        if not trajectory.steps:
            return {}
        
        # Analyze action duration trends
        durations = [step.result.duration for step in trajectory.steps]
        
        # Simple trend analysis
        if len(durations) >= 2:
            trend = "increasing" if durations[-1] > durations[0] else "decreasing"
        else:
            trend = "stable"
        
        return {
            "duration_trend": trend,
            "slowest_actions": sorted(
                [(step.action.type, step.result.duration) for step in trajectory.steps],
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }
    
    def _analyze_success_factors(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Analyze factors that contributed to success or failure"""
        if not trajectory.episode_result:
            return {}
        
        factors = []
        
        # Test success rate
        test_results = trajectory.test_results
        if test_results:
            passed = sum(1 for test in test_results if test.success)
            total = len(test_results)
            factors.append(f"Test success rate: {passed}/{total} ({passed/total*100:.1f}%)")
        
        # Action efficiency
        if trajectory.episode_result.total_actions < 10:
            factors.append("Efficient action usage")
        elif trajectory.episode_result.total_actions > 50:
            factors.append("High action count - may indicate inefficiency")
        
        # Error handling
        error_count = sum(1 for step in trajectory.steps if not step.result.success)
        if error_count == 0:
            factors.append("No errors encountered")
        elif error_count > 5:
            factors.append("Multiple errors encountered")
        
        return {
            "success": trajectory.episode_result.success,
            "factors": factors,
            "final_score": trajectory.episode_result.final_score
        }
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message"""
        error_lower = error_message.lower()
        
        if "timeout" in error_lower:
            return "timeout"
        elif "permission" in error_lower or "access" in error_lower:
            return "permission"
        elif "not found" in error_lower or "no such" in error_lower:
            return "file_not_found"
        elif "syntax" in error_lower:
            return "syntax_error"
        elif "import" in error_lower:
            return "import_error"
        else:
            return "other"
    
    async def suggest_improvements(self, trajectory: Trajectory) -> List[str]:
        """Suggest improvements based on trajectory analysis"""
        suggestions = []
        analysis = await self.analyze_trajectory(trajectory)
        
        # Suggestions based on error rate
        if analysis["error_patterns"]["error_rate"] > 0.3:
            suggestions.append("High error rate detected. Consider improving error handling or simplifying tasks.")
        
        # Suggestions based on action efficiency
        if analysis["total_steps"] > 50:
            suggestions.append("High action count. Consider optimizing the approach or breaking down the task.")
        
        # Suggestions based on test results
        if trajectory.test_results:
            failed_tests = [test for test in trajectory.test_results if not test.success]
            if len(failed_tests) > 0:
                suggestions.append(f"Focus on fixing {len(failed_tests)} failing tests.")
        
        # Suggestions based on performance
        perf_metrics = analysis["performance_metrics"]
        if perf_metrics.get("avg_action_duration", 0) > 10:
            suggestions.append("Actions taking too long. Consider optimizing commands or increasing timeout.")
        
        return suggestions


class SafetyFilterPlugin(ActionFilterPlugin):
    """Plugin that filters potentially unsafe actions"""
    
    @property
    def name(self) -> str:
        return "safety_filter"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_hooks(self) -> List[PluginHookType]:
        return [PluginHookType.PRE_ACTION]
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.blocked_commands = config.get("blocked_commands", [
            "rm -rf /",
            "format",
            "shutdown",
            "reboot",
            "halt",
            "poweroff"
        ])
        self.blocked_patterns = config.get("blocked_patterns", [
            r"rm\s+-rf\s+/",
            r"sudo\s+.*",
            r"chmod\s+777\s+/",
        ])
    
    async def execute(self, hook: PluginHookType, context: PluginContext) -> PluginContext:
        """Execute plugin logic"""
        if hook == PluginHookType.PRE_ACTION:
            actions = context.get("actions", [])
            filtered_actions = await self.filter_actions(actions, context)
            context.set("actions", filtered_actions)
        
        return context
    
    async def filter_actions(self, actions: List[Action], context: PluginContext) -> List[Action]:
        """Filter potentially unsafe actions"""
        filtered_actions = []
        
        for action in actions:
            if self._is_safe_action(action):
                filtered_actions.append(action)
            else:
                # Log blocked action
                print(f"Blocked unsafe action: {action.type} - {action.content}")
        
        return filtered_actions
    
    def _is_safe_action(self, action: Action) -> bool:
        """Check if an action is safe"""
        if action.type != ActionType.COMMAND:
            return True
        
        command = action.content.lower().strip()
        
        # Check blocked commands
        for blocked_cmd in self.blocked_commands:
            if blocked_cmd.lower() in command:
                return False
        
        # Check blocked patterns
        import re
        for pattern in self.blocked_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False
        
        return True 