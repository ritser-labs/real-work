import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

from .config import (
    FrameworkConfig, EnvironmentConfig, LLMConfig, RolloutConfig,
    Action, ActionResult, ActionType, EpisodeResult, TestResult
)
from .episode_controller import EpisodeController
from .trajectory import TrajectoryManager, Trajectory
from .test_runner import UnitTestRunner
from ..environments.environment import Environment, StateManager
from ..agents.llm_agent import LLMAgent, LLMAgentFactory
from ..plugins.manager import PluginManager
from ..plugins.base import PluginHookType, PluginContext, PluginType


class Episode:
    """Represents a single episode execution"""
    
    def __init__(self, environment: Environment, agent: LLMAgent, trajectory_manager: TrajectoryManager,
                 test_runner: UnitTestRunner, plugin_manager: PluginManager, config: FrameworkConfig):
        self.environment = environment
        self.agent = agent
        self.trajectory_manager = trajectory_manager
        self.test_runner = test_runner
        self.plugin_manager = plugin_manager
        self.config = config
        self.episode_id = str(uuid.uuid4())
        self.trajectory_id = None
        self.start_time = None
        self.end_time = None
        self.actions_taken = 0
        self.logger = logging.getLogger(f"Episode[{self.episode_id[:8]}]")
        
    async def run(self) -> EpisodeResult:
        """Run a complete episode"""
        self.start_time = time.time()
        self.logger.info(f"Starting episode for environment {self.environment.config.id}")
        
        try:
            # Initialize trajectory
            self.trajectory_id = self.trajectory_manager.create_trajectory(
                self.environment.config.id,
                metadata={"episode_id": self.episode_id}
            )
            
            # Initialize LLM conversation
            await self.agent.initialize_conversation(
                self.config.template_prompt,
                self.environment.config.prompt
            )
            
            # Execute pre-episode plugins
            context = PluginContext(
                episode_id=self.episode_id,
                environment_id=self.environment.config.id,
                trajectory_id=self.trajectory_id
            )
            
            context = await self.plugin_manager.execute_hook(
                PluginHookType.PRE_EPISODE, context
            )
            
            # Main episode loop
            result = await self._execute_episode_loop()
            
            # Run final tests
            test_results = await self.test_runner.run_tests(self.environment)
            await self.trajectory_manager.add_test_results(self.trajectory_id, test_results)
            
            # Calculate final score
            final_score = self.test_runner.calculate_test_score(test_results)
            
            # Get token usage statistics from the agent
            token_usage = self.agent.get_token_usage_stats() if hasattr(self.agent, 'get_token_usage_stats') else None
            
            # Create episode result
            episode_result = EpisodeResult(
                environment_id=self.environment.config.id,
                success=result["success"],
                total_actions=self.actions_taken,
                duration=time.time() - self.start_time,
                test_results=test_results,
                terminated_reason=result["reason"],
                final_score=final_score,
                token_usage=token_usage
            )
            
            # Finalize trajectory
            await self.trajectory_manager.finalize_trajectory(self.trajectory_id, episode_result)
            
            # Execute post-episode plugins
            context.update({
                "episode_result": episode_result,
                "trajectory": self.trajectory_manager.get_trajectory(self.trajectory_id),
                "test_results": test_results
            })
            
            context = await self.plugin_manager.execute_hook(
                PluginHookType.POST_EPISODE, context
            )
            
            self.logger.info(f"Episode completed: {episode_result.terminated_reason}, Score: {final_score:.2f}")
            
            return episode_result
            
        except Exception as e:
            self.logger.error(f"Error during episode execution: {e}")
            
            # Create failed episode result
            episode_result = EpisodeResult(
                environment_id=self.environment.config.id,
                success=False,
                total_actions=self.actions_taken,
                duration=time.time() - self.start_time if self.start_time else 0,
                test_results=[],
                terminated_reason=f"error: {str(e)}",
                final_score=0.0
            )
            
            if self.trajectory_id:
                await self.trajectory_manager.finalize_trajectory(self.trajectory_id, episode_result)
            
            return episode_result
    
    async def _execute_episode_loop(self) -> Dict[str, Any]:
        """Execute the main episode loop"""
        timeout = self.config.timeout_config.global_timeout
        start_time = time.time()
        
        previous_result = None
        
        while True:
            # Check global timeout
            if time.time() - start_time > timeout:
                self.logger.error(f"Global timeout reached after {timeout} seconds")
                return {"success": False, "reason": "global_timeout"}
            
            # Check action limit
            if self.actions_taken >= 100:  # Safety limit
                self.logger.error(f"Action limit reached: {self.actions_taken}")
                return {"success": False, "reason": "action_limit"}
            
            self.logger.info(f"Episode loop iteration {self.actions_taken + 1}, elapsed: {time.time() - start_time:.1f}s")
            
            # Get next actions from LLM
            try:
                self.logger.debug("Calling agent.get_next_action...")
                result = await self.agent.get_next_action(previous_result)
                if isinstance(result, tuple):
                    actions, tool_calls_info = result
                else:
                    # Handle backwards compatibility
                    actions = result
                    tool_calls_info = []
                
                self.logger.info(f"Got {len(actions)} actions from LLM")  # type: ignore
                for i, action in enumerate(actions):  # type: ignore
                    self.logger.debug(f"Action {i}: {action.type} - {action.content[:100]}...")
                    
            except Exception as e:
                self.logger.error(f"Error getting next action: {e}")
                return {"success": False, "reason": f"llm_error: {e}"}
            
            # Check if we got valid actions
            if not actions:
                self.logger.info("No actions received from LLM - treating as task completion")
                return {"success": True, "reason": "completed_naturally"}
            
            # Execute pre-action plugins
            context = PluginContext(
                episode_id=self.episode_id,
                environment_id=self.environment.config.id,
                trajectory_id=self.trajectory_id,
                actions=actions
            )
            
            context = await self.plugin_manager.execute_hook(
                PluginHookType.PRE_ACTION, context
            )
            
            # Get filtered actions
            filtered_actions = context.get("actions", actions)
            if not filtered_actions:
                self.logger.info("No actions after plugin filtering - treating as task completion")
                return {"success": True, "reason": "completed_naturally"}
            
            self.logger.debug(f"After plugin filtering: {len(filtered_actions)} actions")
            
            # Execute actions
            for i, action in enumerate(filtered_actions):
                self.logger.info(f"Executing action {i+1}/{len(filtered_actions)}: {action.type}")
                
                if action.type == ActionType.DONE:
                    self.logger.info("LLM called mark_done - episode completed")
                    return {"success": True, "reason": "completed"}
                
                # Execute action
                self.logger.debug(f"Environment executing: {action.content[:100]}...")
                result = await self.environment.execute_action(action)
                self.actions_taken += 1
                
                self.logger.info(f"Action result: success={result.success}, duration={result.duration:.2f}s")
                if not result.success:
                    self.logger.warning(f"Action failed: {result.error[:200]}...")
                else:
                    self.logger.debug(f"Action output: {result.output[:200]}...")
                
                # Save state snapshot periodically
                state_snapshot = None
                if self.config.rollout_config.state_persistence_enabled and self.actions_taken % 5 == 0:
                    snapshot_id = f"{self.episode_id}_{self.actions_taken}"
                    state_snapshot = await self.environment.save_state(snapshot_id)
                
                # Add to trajectory
                if self.trajectory_id:
                    await self.trajectory_manager.add_step(
                        self.trajectory_id, action, result, state_snapshot
                    )
                
                # Execute post-action plugins
                context = PluginContext(
                    episode_id=self.episode_id,
                    environment_id=self.environment.config.id,
                    trajectory_id=self.trajectory_id,
                    action=action,
                    result=result
                )
                
                context = await self.plugin_manager.execute_hook(
                    PluginHookType.POST_ACTION, context
                )
                
                previous_result = result
                
                # Log action result
                self.logger.debug(f"Action {action.type}: {result.success}")
                
                # Check if we should continue
                if not result.success and "timeout" in result.error.lower():
                    self.logger.error("Action timed out")
                    return {"success": False, "reason": "action_timeout"}
        
        return {"success": False, "reason": "unknown"}


class RolloutManager:
    """Manages the execution of multiple rollouts with parallel support"""
    
    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.trajectory_manager = TrajectoryManager(
            config.rollout_config.trajectory_output_path,
            config.rollout_config.save_trajectory_interval
        )
        self.test_runner = UnitTestRunner(config.timeout_config)
        self.plugin_manager = PluginManager()
        self.state_manager = StateManager()
        self.episode_controller = EpisodeController(config.episode_control_config)
        self.logger = logging.getLogger("RolloutManager")
        
        # Track rollout statistics
        self.rollout_stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "failed_episodes": 0,
            "total_duration": 0.0,
            "environments": {}
        }
    
    async def initialize(self) -> bool:
        """Initialize the rollout manager"""
        try:
            # Load plugins
            if self.config.rollout_config.enable_plugins:
                await self._load_plugins()
            
            self.logger.info("RolloutManager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing RolloutManager: {e}")
            return False
    
    async def _load_plugins(self) -> None:
        """Load plugins specified in configuration"""
        # Load built-in plugins
        from ..plugins.builtin import (
            PerformanceBasedRolloutPlugin,
            TrajectoryAnalysisPlugin,
            SafetyFilterPlugin
        )
        
        # Load default plugins
        await self.plugin_manager.load_plugin(PerformanceBasedRolloutPlugin)
        await self.plugin_manager.load_plugin(TrajectoryAnalysisPlugin)
        await self.plugin_manager.load_plugin(SafetyFilterPlugin)
        
        # Load additional plugins from config
        for plugin_module in self.config.plugins:
            await self.plugin_manager.load_plugins_from_module(plugin_module)
    
    async def run_rollouts(self) -> List[EpisodeResult]:
        """Run rollouts for all environments"""
        all_results = []
        
        # Execute pre-rollout plugins
        context = PluginContext(
            environments=[env.id for env in self.config.environments],
            config=self.config
        )
        
        context = await self.plugin_manager.execute_hook(
            PluginHookType.PRE_ROLLOUT, context
        )
        
        # Run rollouts in parallel
        if self.config.rollout_config.max_parallel_rollouts > 1:
            results = await self._run_parallel_rollouts()
        else:
            results = await self._run_sequential_rollouts()
        
        all_results.extend(results)
        
        # Execute post-rollout plugins
        context = PluginContext(
            results=all_results,
            rollout_stats=self.rollout_stats
        )
        
        context = await self.plugin_manager.execute_hook(
            PluginHookType.POST_ROLLOUT, context
        )
        
        # Save all trajectories
        await self.trajectory_manager.save_all_trajectories()
        
        return all_results
    
    async def _run_parallel_rollouts(self) -> List[EpisodeResult]:
        """Run rollouts in parallel"""
        semaphore = asyncio.Semaphore(self.config.rollout_config.max_parallel_rollouts)
        
        async def run_with_semaphore(env_config: EnvironmentConfig) -> List[EpisodeResult]:
            async with semaphore:
                return await self._run_environment_rollouts(env_config)

        tasks = [run_with_semaphore(env_config) for env_config in self.config.environments]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        all_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Error in parallel rollout: {result}")
            elif isinstance(result, list):
                all_results.extend(result)
            else:
                self.logger.warning(f"Unexpected result type: {type(result)}")
        
        return all_results
    
    async def _run_sequential_rollouts(self) -> List[EpisodeResult]:
        """Run rollouts sequentially"""
        all_results = []
        
        for env_config in self.config.environments:
            results = await self._run_environment_rollouts(env_config)
            all_results.extend(results)
        
        return all_results
    
    async def _run_environment_rollouts(self, env_config: EnvironmentConfig) -> List[EpisodeResult]:
        """Run rollouts for a specific environment"""
        results = []
        
        self.logger.info(f"Starting rollouts for environment {env_config.id}")
        
        # Initialize environment
        environment = Environment(
            env_config,
            self.config.timeout_config,
            self.state_manager
        )
        
        if not await environment.initialize():
            self.logger.error(f"Failed to initialize environment {env_config.id}")
            return results
        
        try:
            episode_count = 0
            
            # Run episodes until episode controller or plugins say to stop
            while True:
                # Check episode controller first
                controller_decision = self.episode_controller.should_continue_episodes(
                    episode_count, results, env_config.id
                )
                
                if not controller_decision["continue"]:
                    self.logger.info(f"Episode controller stopping rollouts for {env_config.id}: {controller_decision['reason']}")
                    break
                
                # Check if we should continue with plugins
                context = PluginContext(
                    environment_id=env_config.id,
                    episode_count=episode_count,
                    results=results
                )
                
                should_continue = True
                rollout_plugins = self.plugin_manager.get_plugins_by_type(PluginType.ROLLOUT_MANAGER)
                
                for plugin in rollout_plugins:
                    if plugin.is_enabled() and hasattr(plugin, 'should_continue_rollout'):
                        try:
                            plugin_should_continue = await getattr(plugin, 'should_continue_rollout')(context)
                            should_continue = should_continue and plugin_should_continue
                        except AttributeError:
                            pass  # Plugin doesn't have should_continue_rollout method
                
                if not should_continue:
                    self.logger.info(f"Stopping rollouts for {env_config.id} based on plugin decision")
                    break
                
                # Create LLM agent
                if not self.config.llm_config:
                    self.logger.error("LLM config is None")
                    return []
                agent = LLMAgentFactory.create_agent(self.config.llm_config)
                
                # Run episode
                episode = Episode(
                    environment, agent, self.trajectory_manager,
                    self.test_runner, self.plugin_manager, self.config
                )
                
                episode_result = await episode.run()
                results.append(episode_result)
                
                # Update statistics
                self._update_stats(episode_result)
                
                episode_count += 1
                
                self.logger.info(f"Episode {episode_count} completed for {env_config.id}")
        
        finally:
            # Cleanup environment
            await environment.cleanup()
        
        return results
    
    def _update_stats(self, episode_result: EpisodeResult) -> None:
        """Update rollout statistics"""
        self.rollout_stats["total_episodes"] += 1
        
        if episode_result.success:
            self.rollout_stats["successful_episodes"] += 1
        else:
            self.rollout_stats["failed_episodes"] += 1
        
        self.rollout_stats["total_duration"] += episode_result.duration
        
        # Update environment-specific stats
        env_id = episode_result.environment_id
        if env_id not in self.rollout_stats["environments"]:
            self.rollout_stats["environments"][env_id] = {
                "episodes": 0,
                "successful": 0,
                "failed": 0,
                "total_duration": 0.0,
                "avg_score": 0.0
            }
        
        env_stats = self.rollout_stats["environments"][env_id]
        env_stats["episodes"] += 1
        env_stats["total_duration"] += episode_result.duration
        
        if episode_result.success:
            env_stats["successful"] += 1
        else:
            env_stats["failed"] += 1
        
        # Update average score
        env_stats["avg_score"] = (env_stats["avg_score"] * (env_stats["episodes"] - 1) + 
                                 episode_result.final_score) / env_stats["episodes"]
    
    async def export_results(self, output_file: str) -> None:
        """Export all results to a file"""
        await self.trajectory_manager.export_trajectories(output_file)
        
        # Also export statistics
        stats_file = Path(output_file).with_suffix('.stats.json')
        
        import json
        with open(stats_file, 'w') as f:
            json.dump(self.rollout_stats, f, indent=2)
        
        self.logger.info(f"Results exported to {output_file}")
    
    def get_rollout_summary(self) -> Dict[str, Any]:
        """Get summary of rollout execution"""
        if self.rollout_stats["total_episodes"] == 0:
            return {"message": "No episodes executed"}
        
        success_rate = (self.rollout_stats["successful_episodes"] / 
                       self.rollout_stats["total_episodes"])
        
        avg_duration = (self.rollout_stats["total_duration"] / 
                       self.rollout_stats["total_episodes"])
        
        return {
            "total_episodes": self.rollout_stats["total_episodes"],
            "success_rate": success_rate,
            "avg_duration": avg_duration,
            "total_duration": self.rollout_stats["total_duration"],
            "environments": self.rollout_stats["environments"],
            "plugin_stats": self.plugin_manager.get_plugin_statistics()
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            # Save trajectories first
            await self.trajectory_manager.save_all_trajectories()
            
            # Unload plugins
            await self.plugin_manager.unload_all_plugins()
            
            # Force cleanup any remaining Docker containers
            await self._force_cleanup_containers()
            
            self.logger.info("RolloutManager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def _force_cleanup_containers(self) -> None:
        """Force cleanup any remaining Docker containers that might be orphaned"""
        try:
            import subprocess
            import docker
            
            # Get list of containers that might be from our framework
            client = docker.from_env()
            containers = client.containers.list(all=True)
            
            cleanup_count = 0
            for container in containers:
                try:
                    # Check if this looks like one of our containers
                    # Our containers typically have specific labels or names
                    container_info = container.attrs
                    labels = container_info.get('Config', {}).get('Labels', {})
                    
                    # Look for containers that might be orphaned from our framework
                    # This is a best-effort cleanup
                    if (container.status in ['running', 'created'] and 
                        any(keyword in str(container_info).lower() for keyword in 
                            ['workspace', 'tmp', 'python', 'tail -f /dev/null'])):
                        
                        self.logger.debug(f"Cleaning up potentially orphaned container: {container.id[:12]}")
                        container.stop(timeout=5)
                        container.remove(force=True)
                        cleanup_count += 1
                        
                except Exception as e:
                    self.logger.debug(f"Error cleaning up container {container.id[:12]}: {e}")
            
            if cleanup_count > 0:
                self.logger.info(f"Cleaned up {cleanup_count} potentially orphaned containers")
                
        except Exception as e:
            self.logger.debug(f"Error during force container cleanup: {e}") 