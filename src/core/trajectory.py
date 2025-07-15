import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import asyncio
import aiofiles
from pathlib import Path

from .config import Action, ActionResult, TestResult, EpisodeResult, EnvironmentConfig


@dataclass
class TrajectoryStep:
    """Represents a single step in a trajectory"""
    step_id: str
    timestamp: str
    action: Action
    result: ActionResult
    state_snapshot: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'step_id': self.step_id,
            'timestamp': self.timestamp,
            'action': self.action.model_dump(),
            'result': self.result.model_dump(),
            'state_snapshot': self.state_snapshot
        }


@dataclass
class Trajectory:
    """Represents a complete trajectory for an episode"""
    trajectory_id: str
    environment_id: str
    start_time: str
    end_time: Optional[str] = None
    steps: List[TrajectoryStep] = None
    test_results: List[TestResult] = None
    episode_result: Optional[EpisodeResult] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.test_results is None:
            self.test_results = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'trajectory_id': self.trajectory_id,
            'environment_id': self.environment_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'steps': [step.to_dict() for step in self.steps],
            'test_results': [result.model_dump() for result in self.test_results],
            'episode_result': self.episode_result.model_dump() if self.episode_result else None,
            'metadata': self.metadata
        }


class TrajectoryManager:
    """Manages trajectory tracking and persistence"""
    
    def __init__(self, output_path: str = "trajectories", save_interval: int = 10):
        self.output_path = Path(output_path)
        self.save_interval = save_interval
        self.trajectories: Dict[str, Trajectory] = {}
        self.step_counts: Dict[str, int] = {}
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def create_trajectory(self, environment_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new trajectory"""
        trajectory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            environment_id=environment_id,
            start_time=timestamp,
            metadata=metadata or {}
        )
        
        self.trajectories[trajectory_id] = trajectory
        self.step_counts[trajectory_id] = 0
        
        return trajectory_id
    
    async def add_step(self, trajectory_id: str, action: Action, result: ActionResult, 
                      state_snapshot: Optional[Dict[str, Any]] = None) -> None:
        """Add a step to a trajectory"""
        if trajectory_id not in self.trajectories:
            raise ValueError(f"Trajectory {trajectory_id} not found")
        
        step_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        step = TrajectoryStep(
            step_id=step_id,
            timestamp=timestamp,
            action=action,
            result=result,
            state_snapshot=state_snapshot
        )
        
        self.trajectories[trajectory_id].steps.append(step)
        self.step_counts[trajectory_id] += 1
        
        # Auto-save at intervals
        if self.step_counts[trajectory_id] % self.save_interval == 0:
            await self._save_trajectory(trajectory_id)
    
    async def add_test_results(self, trajectory_id: str, test_results: List[TestResult]) -> None:
        """Add test results to a trajectory"""
        if trajectory_id not in self.trajectories:
            raise ValueError(f"Trajectory {trajectory_id} not found")
        
        self.trajectories[trajectory_id].test_results.extend(test_results)
        await self._save_trajectory(trajectory_id)
    
    async def finalize_trajectory(self, trajectory_id: str, episode_result: EpisodeResult) -> None:
        """Finalize a trajectory with episode results"""
        if trajectory_id not in self.trajectories:
            raise ValueError(f"Trajectory {trajectory_id} not found")
        
        trajectory = self.trajectories[trajectory_id]
        trajectory.end_time = datetime.now().isoformat()
        trajectory.episode_result = episode_result
        
        await self._save_trajectory(trajectory_id)
    
    async def _save_trajectory(self, trajectory_id: str) -> None:
        """Save trajectory to disk"""
        if trajectory_id not in self.trajectories:
            return
        
        trajectory = self.trajectories[trajectory_id]
        filename = f"{trajectory_id}.json"
        filepath = self.output_path / filename
        
        try:
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(trajectory.to_dict(), indent=2))
        except Exception as e:
            print(f"Error saving trajectory {trajectory_id}: {e}")
    
    async def save_all_trajectories(self) -> None:
        """Save all trajectories to disk"""
        tasks = []
        for trajectory_id in self.trajectories:
            tasks.append(self._save_trajectory(trajectory_id))
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def load_trajectory(self, trajectory_id: str) -> Optional[Trajectory]:
        """Load a trajectory from disk"""
        filename = f"{trajectory_id}.json"
        filepath = self.output_path / filename
        
        if not filepath.exists():
            return None
        
        try:
            async with aiofiles.open(filepath, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                
                # Reconstruct trajectory from dict
                trajectory = Trajectory(
                    trajectory_id=data['trajectory_id'],
                    environment_id=data['environment_id'],
                    start_time=data['start_time'],
                    end_time=data.get('end_time'),
                    metadata=data.get('metadata', {})
                )
                
                # Reconstruct steps
                for step_data in data.get('steps', []):
                    step = TrajectoryStep(
                        step_id=step_data['step_id'],
                        timestamp=step_data['timestamp'],
                        action=Action(**step_data['action']),
                        result=ActionResult(**step_data['result']),
                        state_snapshot=step_data.get('state_snapshot')
                    )
                    trajectory.steps.append(step)
                
                # Reconstruct test results
                for test_data in data.get('test_results', []):
                    test_result = TestResult(**test_data)
                    trajectory.test_results.append(test_result)
                
                # Reconstruct episode result
                if data.get('episode_result'):
                    trajectory.episode_result = EpisodeResult(**data['episode_result'])
                
                return trajectory
                
        except Exception as e:
            print(f"Error loading trajectory {trajectory_id}: {e}")
            return None
    
    def get_trajectory(self, trajectory_id: str) -> Optional[Trajectory]:
        """Get a trajectory from memory"""
        return self.trajectories.get(trajectory_id)
    
    def list_trajectories(self) -> List[str]:
        """List all trajectory IDs"""
        return list(self.trajectories.keys())
    
    async def export_trajectories(self, output_file: str) -> None:
        """Export all trajectories to a single JSON file"""
        all_trajectories = []
        for trajectory in self.trajectories.values():
            all_trajectories.append(trajectory.to_dict())
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            async with aiofiles.open(output_path, 'w') as f:
                await f.write(json.dumps(all_trajectories, indent=2))
        except Exception as e:
            print(f"Error exporting trajectories: {e}")
    
    def get_trajectory_summary(self, trajectory_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a trajectory"""
        trajectory = self.trajectories.get(trajectory_id)
        if not trajectory:
            return None
        
        return {
            'trajectory_id': trajectory_id,
            'environment_id': trajectory.environment_id,
            'start_time': trajectory.start_time,
            'end_time': trajectory.end_time,
            'total_steps': len(trajectory.steps),
            'total_tests': len(trajectory.test_results),
            'success': trajectory.episode_result.success if trajectory.episode_result else False,
            'final_score': trajectory.episode_result.final_score if trajectory.episode_result else 0.0,
            'terminated_reason': trajectory.episode_result.terminated_reason if trajectory.episode_result else None
        } 