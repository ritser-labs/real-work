import importlib
import inspect
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import logging

from .base import BasePlugin, PluginType, PluginHookType, PluginContext, PluginResult


class PluginManager:
    """Manages plugin loading, initialization, and execution"""
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugins_by_type: Dict[PluginType, List[BasePlugin]] = {}
        self.plugins_by_hook: Dict[PluginHookType, List[BasePlugin]] = {}
        self.logger = logging.getLogger("PluginManager")
        
        # Initialize plugin type mapping
        for plugin_type in PluginType:
            self.plugins_by_type[plugin_type] = []
        
        # Initialize hook mapping
        for hook_type in PluginHookType:
            self.plugins_by_hook[hook_type] = []
    
    async def load_plugin(self, plugin_class: Type[BasePlugin], config: Dict[str, Any] = None) -> bool:
        """Load a plugin class"""
        try:
            # Create plugin instance
            plugin = plugin_class(config or {})
            
            # Initialize plugin
            if not await plugin.initialize():
                self.logger.error(f"Failed to initialize plugin {plugin.name}")
                return False
            
            # Register plugin
            self.plugins[plugin.name] = plugin
            
            # Add to type mapping
            self.plugins_by_type[plugin.plugin_type].append(plugin)
            
            # Add to hook mapping
            for hook in plugin.supported_hooks:
                self.plugins_by_hook[hook].append(plugin)
            
            # Sort by priority (higher priority first)
            self.plugins_by_hook[hook].sort(key=lambda p: p.priority, reverse=True)
            
            self.logger.info(f"Loaded plugin: {plugin.name} v{plugin.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading plugin: {e}")
            return False
    
    async def load_plugins_from_module(self, module_path: str, config: Dict[str, Any] = None) -> int:
        """Load plugins from a module"""
        try:
            module = importlib.import_module(module_path)
            loaded_count = 0
            
            # Find all plugin classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj is not BasePlugin and
                    not inspect.isabstract(obj)):
                    
                    if await self.load_plugin(obj, config):
                        loaded_count += 1
            
            return loaded_count
            
        except Exception as e:
            self.logger.error(f"Error loading plugins from module {module_path}: {e}")
            return 0
    
    async def load_plugins_from_directory(self, directory: str, config: Dict[str, Any] = None) -> int:
        """Load plugins from a directory"""
        plugin_dir = Path(directory)
        if not plugin_dir.exists():
            self.logger.warning(f"Plugin directory {directory} does not exist")
            return 0
        
        loaded_count = 0
        
        # Find all Python files in the directory
        for python_file in plugin_dir.glob("*.py"):
            if python_file.name.startswith("_"):
                continue
            
            try:
                # Convert file path to module path
                module_name = python_file.stem
                module_path = f"{directory.replace('/', '.')}.{module_name}"
                
                count = await self.load_plugins_from_module(module_path, config)
                loaded_count += count
                
            except Exception as e:
                self.logger.error(f"Error loading plugin from {python_file}: {e}")
        
        return loaded_count
    
    async def execute_hook(self, hook: PluginHookType, context: PluginContext) -> PluginContext:
        """Execute all plugins for a specific hook"""
        if hook not in self.plugins_by_hook:
            return context
        
        plugins = [p for p in self.plugins_by_hook[hook] if p.is_enabled()]
        
        for plugin in plugins:
            try:
                self.logger.debug(f"Executing plugin {plugin.name} for hook {hook}")
                context = await plugin.execute(hook, context)
                
            except Exception as e:
                self.logger.error(f"Error executing plugin {plugin.name} for hook {hook}: {e}")
        
        return context
    
    async def execute_plugins_by_type(self, plugin_type: PluginType, method_name: str, *args, **kwargs) -> List[Any]:
        """Execute a specific method on all plugins of a given type"""
        results = []
        
        if plugin_type not in self.plugins_by_type:
            return results
        
        plugins = [p for p in self.plugins_by_type[plugin_type] if p.is_enabled()]
        
        for plugin in plugins:
            try:
                if hasattr(plugin, method_name):
                    method = getattr(plugin, method_name)
                    if callable(method):
                        result = await method(*args, **kwargs)
                        results.append(result)
                        
            except Exception as e:
                self.logger.error(f"Error executing {method_name} on plugin {plugin.name}: {e}")
        
        return results
    
    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get a plugin by name"""
        return self.plugins.get(name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all plugins of a specific type"""
        return self.plugins_by_type.get(plugin_type, [])
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins"""
        return [
            {
                "name": plugin.name,
                "version": plugin.version,
                "type": plugin.plugin_type,
                "enabled": plugin.is_enabled(),
                "priority": plugin.priority,
                "hooks": plugin.supported_hooks
            }
            for plugin in self.plugins.values()
        ]
    
    async def enable_plugin(self, name: str) -> bool:
        """Enable a plugin"""
        plugin = self.plugins.get(name)
        if plugin:
            plugin.enable()
            self.logger.info(f"Enabled plugin: {name}")
            return True
        return False
    
    async def disable_plugin(self, name: str) -> bool:
        """Disable a plugin"""
        plugin = self.plugins.get(name)
        if plugin:
            plugin.disable()
            self.logger.info(f"Disabled plugin: {name}")
            return True
        return False
    
    async def unload_plugin(self, name: str) -> bool:
        """Unload a plugin"""
        plugin = self.plugins.get(name)
        if not plugin:
            return False
        
        try:
            # Cleanup plugin
            await plugin.cleanup()
            
            # Remove from mappings
            del self.plugins[name]
            
            # Remove from type mapping
            if plugin.plugin_type in self.plugins_by_type:
                self.plugins_by_type[plugin.plugin_type].remove(plugin)
            
            # Remove from hook mapping
            for hook in plugin.supported_hooks:
                if hook in self.plugins_by_hook:
                    self.plugins_by_hook[hook].remove(plugin)
            
            self.logger.info(f"Unloaded plugin: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unloading plugin {name}: {e}")
            return False
    
    async def unload_all_plugins(self) -> None:
        """Unload all plugins"""
        plugin_names = list(self.plugins.keys())
        
        for name in plugin_names:
            await self.unload_plugin(name)
    
    async def reload_plugin(self, name: str) -> bool:
        """Reload a plugin"""
        plugin = self.plugins.get(name)
        if not plugin:
            return False
        
        plugin_class = type(plugin)
        config = plugin.config
        
        # Unload current plugin
        if not await self.unload_plugin(name):
            return False
        
        # Reload plugin
        return await self.load_plugin(plugin_class, config)
    
    def get_plugin_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded plugins"""
        stats = {
            "total_plugins": len(self.plugins),
            "enabled_plugins": len([p for p in self.plugins.values() if p.is_enabled()]),
            "disabled_plugins": len([p for p in self.plugins.values() if not p.is_enabled()]),
            "plugins_by_type": {}
        }
        
        for plugin_type in PluginType:
            plugins = self.plugins_by_type[plugin_type]
            stats["plugins_by_type"][plugin_type] = {
                "count": len(plugins),
                "enabled": len([p for p in plugins if p.is_enabled()]),
                "disabled": len([p for p in plugins if not p.is_enabled()])
            }
        
        return stats 