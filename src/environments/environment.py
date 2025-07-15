import docker
import asyncio
import time
import json
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import io
import logging

from ..core.config import EnvironmentConfig, Action, ActionResult, ActionType, TimeoutConfig


class StateManager:
    """Manages file system state persistence for environments"""
    
    def __init__(self, state_path: str = "state_snapshots"):
        self.state_path = Path(state_path)
        self.state_path.mkdir(parents=True, exist_ok=True)
        
    async def save_state(self, container_id: str, snapshot_id: str, working_dir: str = "/workspace") -> Dict[str, Any]:
        """Save the current state of a container's file system"""
        try:
            client = docker.from_env()
            container = client.containers.get(container_id)
            
            # Create tar archive of the working directory
            snapshot_path = self.state_path / f"{snapshot_id}.tar"
            
            # Get the archive from the container
            archive_data, _ = container.get_archive(working_dir)
            
            # Save to file
            with open(snapshot_path, 'wb') as f:
                for chunk in archive_data:
                    f.write(chunk)
            
            # Get basic stats
            stats = {
                'snapshot_id': snapshot_id,
                'container_id': container_id,
                'working_dir': working_dir,
                'timestamp': datetime.now().isoformat(),
                'size_bytes': snapshot_path.stat().st_size
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error saving state: {e}")
            return {'error': str(e)}
    
    async def restore_state(self, container_id: str, snapshot_id: str, working_dir: str = "/workspace") -> bool:
        """Restore a container's file system from a snapshot"""
        try:
            client = docker.from_env()
            container = client.containers.get(container_id)
            
            snapshot_path = self.state_path / f"{snapshot_id}.tar"
            
            if not snapshot_path.exists():
                logging.error(f"Snapshot {snapshot_id} not found")
                return False
            
            # Read the archive
            with open(snapshot_path, 'rb') as f:
                archive_data = f.read()
            
            # Restore to container
            container.put_archive(working_dir, archive_data)
            
            return True
            
        except Exception as e:
            logging.error(f"Error restoring state: {e}")
            return False


class Environment:
    """Manages a Docker environment for LLM agent execution"""
    
    def __init__(self, config: EnvironmentConfig, timeout_config: TimeoutConfig, 
                 state_manager: Optional[StateManager] = None):
        self.config = config
        self.timeout_config = timeout_config
        self.state_manager = state_manager or StateManager()
        self.container = None
        self.client = None
        self.is_initialized = False
        self.logger = logging.getLogger(f"Environment[{config.id}]")
        
    async def initialize(self) -> bool:
        """Initialize the Docker environment"""
        try:
            self.client = docker.from_env()
            
            # Create container
            self.container = self.client.containers.run(
                image=self.config.docker_image,
                command="tail -f /dev/null",  # Keep container running
                working_dir=self.config.working_directory,
                environment=self.config.environment_variables,
                detach=True,
                user="1000:1000",  # Non-root user for security
                network_mode="bridge",
                mem_limit="2g",  # Memory limit
                cpu_count=2,  # CPU limit
                remove=True  # Auto-remove when stopped
            )
            
            # Wait for container to be ready
            await asyncio.sleep(2)
            
            # Run initialization commands
            for cmd in self.config.init_commands:
                result = await self._execute_command(cmd, timeout=self.timeout_config.command_timeout)
                if not result.success:
                    self.logger.error(f"Init command failed: {cmd}")
                    return False
            
            self.is_initialized = True
            self.logger.info(f"Environment {self.config.id} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing environment: {e}")
            return False
    
    async def execute_action(self, action: Action) -> ActionResult:
        """Execute an action in the environment"""
        if not self.is_initialized:
            return ActionResult(
                success=False,
                error="Environment not initialized",
                duration=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        start_time = time.time()
        
        try:
            if action.type == ActionType.COMMAND:
                return await self._execute_command(
                    action.content,
                    timeout=action.timeout or self.timeout_config.command_timeout,
                    working_dir=action.working_directory
                )
            elif action.type == ActionType.FILE_WRITE:
                return await self._write_file(action.content)
            elif action.type == ActionType.FILE_READ:
                return await self._read_file(action.content)
            elif action.type == ActionType.DONE:
                return ActionResult(
                    success=True,
                    output="Episode completed",
                    duration=time.time() - start_time,
                    timestamp=datetime.now().isoformat()
                )
            else:
                return ActionResult(
                    success=False,
                    error=f"Unknown action type: {action.type}",
                    duration=time.time() - start_time,
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            self.logger.error(f"Error executing action: {e}")
            return ActionResult(
                success=False,
                error=str(e),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
    
    async def _execute_command(self, command: str, timeout: int = None, 
                              working_dir: str = None) -> ActionResult:
        """Execute a shell command in the container"""
        start_time = time.time()
        
        try:
            if working_dir:
                command = f"cd {working_dir} && {command}"
            
            # Execute command
            exec_result = self.container.exec_run(
                command,
                workdir=working_dir or self.config.working_directory,
                user="1000:1000",
                environment=self.config.environment_variables,
                stdout=True,
                stderr=True,
                detach=False
            )
            
            output = exec_result.output.decode('utf-8', errors='ignore')
            exit_code = exec_result.exit_code
            
            return ActionResult(
                success=exit_code == 0,
                output=output,
                error=output if exit_code != 0 else "",
                exit_code=exit_code,
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                error=str(e),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
    
    async def _write_file(self, content: str) -> ActionResult:
        """Write content to a file in the container"""
        start_time = time.time()
        
        try:
            # Parse the file write request (format: "filepath:content")
            if ':' not in content:
                return ActionResult(
                    success=False,
                    error="Invalid file write format. Use 'filepath:content'",
                    duration=time.time() - start_time,
                    timestamp=datetime.now().isoformat()
                )
            
            filepath, file_content = content.split(':', 1)
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            # Copy to container
            with open(temp_file_path, 'rb') as f:
                data = f.read()
            
            # Create tar archive
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                tarinfo = tarfile.TarInfo(name=Path(filepath).name)
                tarinfo.size = len(data)
                tar.addfile(tarinfo, io.BytesIO(data))
            
            tar_stream.seek(0)
            
            # Put file in container
            success = self.container.put_archive(
                path=str(Path(filepath).parent),
                data=tar_stream.read()
            )
            
            # Clean up temp file
            Path(temp_file_path).unlink()
            
            return ActionResult(
                success=success,
                output=f"File written: {filepath}",
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                error=str(e),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
    
    async def _read_file(self, filepath: str) -> ActionResult:
        """Read a file from the container"""
        start_time = time.time()
        
        try:
            # Get file from container
            archive_data, _ = self.container.get_archive(filepath)
            
            # Extract content
            content = ""
            archive_stream = io.BytesIO()
            for chunk in archive_data:
                archive_stream.write(chunk)
            
            archive_stream.seek(0)
            
            with tarfile.open(fileobj=archive_stream, mode='r') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        file_obj = tar.extractfile(member)
                        content = file_obj.read().decode('utf-8', errors='ignore')
                        break
            
            return ActionResult(
                success=True,
                output=content,
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                error=str(e),
                duration=time.time() - start_time,
                timestamp=datetime.now().isoformat()
            )
    
    async def run_tests(self) -> List[Dict[str, Any]]:
        """Run unit tests and return results"""
        test_results = []
        
        for test_command in self.config.unit_tests:
            start_time = time.time()
            
            try:
                exec_result = self.container.exec_run(
                    test_command,
                    workdir=self.config.working_directory,
                    user="1000:1000",
                    environment=self.config.environment_variables,
                    stdout=True,
                    stderr=True,
                    detach=False
                )
                
                output = exec_result.output.decode('utf-8', errors='ignore')
                exit_code = exec_result.exit_code
                
                test_results.append({
                    'command': test_command,
                    'success': exit_code == 0,
                    'output': output,
                    'error': output if exit_code != 0 else "",
                    'exit_code': exit_code,
                    'duration': time.time() - start_time,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                test_results.append({
                    'command': test_command,
                    'success': False,
                    'output': "",
                    'error': str(e),
                    'exit_code': -1,
                    'duration': time.time() - start_time,
                    'timestamp': datetime.now().isoformat()
                })
        
        return test_results
    
    async def save_state(self, snapshot_id: str) -> Dict[str, Any]:
        """Save current state of the environment"""
        if not self.container:
            return {'error': 'Container not initialized'}
        
        return await self.state_manager.save_state(
            self.container.id,
            snapshot_id,
            self.config.working_directory
        )
    
    async def restore_state(self, snapshot_id: str) -> bool:
        """Restore environment to a previous state"""
        if not self.container:
            return False
        
        return await self.state_manager.restore_state(
            self.container.id,
            snapshot_id,
            self.config.working_directory
        )
    
    async def cleanup(self):
        """Clean up the environment"""
        try:
            if self.container:
                self.container.stop()
                self.container.remove()
                self.logger.info(f"Environment {self.config.id} cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_container_info(self) -> Dict[str, Any]:
        """Get information about the container"""
        if not self.container:
            return {'status': 'not_initialized'}
        
        try:
            self.container.reload()
            return {
                'id': self.container.id,
                'status': self.container.status,
                'image': self.container.image.tags[0] if self.container.image.tags else 'unknown',
                'created': self.container.attrs['Created'],
                'started': self.container.attrs['State'].get('StartedAt'),
                'working_dir': self.config.working_directory
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def get_file_list(self, directory: str = None) -> ActionResult:
        """Get list of files in a directory"""
        dir_path = directory or self.config.working_directory
        return await self._execute_command(f"find {dir_path} -type f -name '*' | head -100")
    
    async def get_directory_structure(self, directory: str = None) -> ActionResult:
        """Get directory structure"""
        dir_path = directory or self.config.working_directory
        return await self._execute_command(f"tree {dir_path} || find {dir_path} -type d | head -50") 