"""
Episode Controller

This module provides a modular and extensible way to control episode execution,
allowing for easy customization of when to stop running episodes.
"""

from typing import List, Optional, Dict, Any
import logging
from abc import ABC, abstractmethod

from .config import EpisodeControlConfig, EpisodeResult


class EpisodeStopCondition(ABC):
    """Abstract base class for episode stop conditions"""
    
    @abstractmethod
    def should_stop(self, episode_count: int, results: List[EpisodeResult], 
                   config: EpisodeControlConfig) -> bool:
        """
        Determine if episode execution should stop
        
        Args:
            episode_count: Number of episodes completed so far
            results: List of episode results
            config: Episode control configuration
            
        Returns:
            True if episodes should stop, False otherwise
        """
        pass
    
    @abstractmethod
    def get_reason(self) -> str:
        """Get the reason why episodes should stop"""
        pass


class MaxEpisodesCondition(EpisodeStopCondition):
    """Stop when maximum number of episodes is reached"""
    
    def should_stop(self, episode_count: int, results: List[EpisodeResult], 
                   config: EpisodeControlConfig) -> bool:
        return episode_count >= config.max_episodes
    
    def get_reason(self) -> str:
        return "max_episodes_reached"


class SuccessCondition(EpisodeStopCondition):
    """Stop when a successful episode occurs (if stop_on_success is enabled)"""
    
    def should_stop(self, episode_count: int, results: List[EpisodeResult], 
                   config: EpisodeControlConfig) -> bool:
        if not config.stop_on_success:
            return False
        
        return any(result.success for result in results)
    
    def get_reason(self) -> str:
        return "success_achieved"


class MinSuccessRateCondition(EpisodeStopCondition):
    """Stop when minimum success rate is achieved (if configured)"""
    
    def should_stop(self, episode_count: int, results: List[EpisodeResult], 
                   config: EpisodeControlConfig) -> bool:
        if config.min_success_rate is None or episode_count < 3:  # Need at least 3 episodes for meaningful rate
            return False
        
        success_count = sum(1 for result in results if result.success)
        success_rate = success_count / len(results)
        
        return success_rate >= config.min_success_rate
    
    def get_reason(self) -> str:
        return "min_success_rate_achieved"


class SafetyLimitCondition(EpisodeStopCondition):
    """Safety limit to prevent infinite loops"""
    
    def should_stop(self, episode_count: int, results: List[EpisodeResult], 
                   config: EpisodeControlConfig) -> bool:
        return episode_count >= config.safety_limit
    
    def get_reason(self) -> str:
        return "safety_limit_reached"


class EpisodeController:
    """Controls episode execution with configurable stop conditions"""
    
    def __init__(self, config: EpisodeControlConfig):
        self.config = config
        self.logger = logging.getLogger("EpisodeController")
        
        # Initialize default stop conditions
        self.stop_conditions = [
            MaxEpisodesCondition(),
            SuccessCondition(),
            MinSuccessRateCondition(),
            SafetyLimitCondition()  # Always include safety limit
        ]
    
    def add_stop_condition(self, condition: EpisodeStopCondition) -> None:
        """Add a custom stop condition"""
        self.stop_conditions.append(condition)
    
    def should_continue_episodes(self, episode_count: int, results: List[EpisodeResult], 
                               environment_id: str) -> Dict[str, Any]:
        """
        Determine if more episodes should be run
        
        Args:
            episode_count: Number of episodes completed so far
            results: List of episode results
            environment_id: ID of the environment
            
        Returns:
            Dictionary with 'continue' (bool) and 'reason' (str) keys
        """
        # Check all stop conditions
        for condition in self.stop_conditions:
            if condition.should_stop(episode_count, results, self.config):
                reason = condition.get_reason()
                self.logger.info(f"Stopping episodes for {environment_id}: {reason}")
                return {
                    "continue": False,
                    "reason": reason,
                    "episode_count": episode_count,
                    "success_count": sum(1 for r in results if r.success),
                    "success_rate": sum(1 for r in results if r.success) / len(results) if results else 0.0
                }
        
        # If no stop conditions are met, continue
        return {
            "continue": True,
            "reason": "continuing",
            "episode_count": episode_count,
            "success_count": sum(1 for r in results if r.success),
            "success_rate": sum(1 for r in results if r.success) / len(results) if results else 0.0
        }
    
    def get_effective_max_episodes(self, environment_id: str) -> int:
        """Get the effective maximum episodes for a specific environment"""
        if self.config.max_episodes_per_env is not None:
            return self.config.max_episodes_per_env
        return self.config.max_episodes
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the episode control configuration"""
        return {
            "max_episodes": self.config.max_episodes,
            "max_episodes_per_env": self.config.max_episodes_per_env,
            "stop_on_success": self.config.stop_on_success,
            "min_success_rate": self.config.min_success_rate,
            "safety_limit": self.config.safety_limit,
            "stop_conditions": len(self.stop_conditions)
        } 