import openai
import json
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from ..core.config import LLMConfig, Action, ActionType, ActionResult


class LLMAgent:
    """LLM agent for interacting with OpenAI API using tool calling"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        
        # Use OpenRouter if no API key provided, otherwise use config
        api_key = config.api_key if config.api_key and config.api_key != "your-api-key-here" else os.getenv("OPENROUTER_API_KEY")
        base_url = config.base_url if config.base_url != "https://api.openai.com/v1" else "https://openrouter.ai/api/v1"
        
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=config.timeout
        )
        
        self.conversation_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("LLMAgent")
        
        # Define tools for the LLM
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute a shell command in the environment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The shell command to execute"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Optional timeout in seconds for the command"
                            },
                            "working_directory": {
                                "type": "string",
                                "description": "Optional working directory to execute the command in"
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["filepath", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read content from a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["filepath"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_done",
                    "description": "Mark the task as completed",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Optional completion message"
                            }
                        },
                        "required": []
                    }
                }
            }
        ]
        
        # Tool mapping for converting tool calls to actions
        self.tool_mapping = {
            "execute_command": self._create_command_action,
            "write_file": self._create_file_write_action,
            "read_file": self._create_file_read_action,
            "mark_done": self._create_done_action
        }
    
    def _create_command_action(self, **kwargs) -> Action:
        """Create a command action from tool call"""
        return Action(
            type=ActionType.COMMAND,
            content=kwargs["command"],
            timeout=kwargs.get("timeout"),
            working_directory=kwargs.get("working_directory")
        )
    
    def _create_file_write_action(self, **kwargs) -> Action:
        """Create a file write action from tool call"""
        return Action(
            type=ActionType.FILE_WRITE,
            content=f"{kwargs['filepath']}:{kwargs['content']}",
            timeout=None,
            working_directory=None
        )
    
    def _create_file_read_action(self, **kwargs) -> Action:
        """Create a file read action from tool call"""
        return Action(
            type=ActionType.FILE_READ,
            content=kwargs["filepath"],
            timeout=None,
            working_directory=None
        )
    
    def _create_done_action(self, **kwargs) -> Action:
        """Create a done action from tool call"""
        return Action(
            type=ActionType.DONE,
            content=kwargs.get("message", "Task completed"),
            timeout=None,
            working_directory=None
        )
    
    async def initialize_conversation(self, template_prompt: str, environment_prompt: str) -> None:
        """Initialize conversation with system prompts"""
        system_prompt = f"""
{template_prompt}

{environment_prompt}

You are working in a Docker environment with shell access. You can execute commands, read/write files, and complete coding tasks. The unit tests will evaluate your work, but you don't have access to the test code itself.

You have access to the following tools:
1. execute_command - Run shell commands
2. write_file - Write content to files
3. read_file - Read content from files
4. mark_done - Mark the task as completed

Always think step by step and be methodical in your approach. When you're done with the task, use the mark_done tool to indicate completion.
"""
        
        self.conversation_history = [
            {"role": "system", "content": system_prompt.strip()}
        ]
        
        self.logger.info("Conversation initialized with system prompt")
    
    async def get_next_action(self, previous_result: Optional[ActionResult] = None) -> List[Action]:
        """Get next action(s) from the LLM using tool calling"""
        try:
            # Add previous result to conversation if available
            if previous_result:
                result_message = f"Previous action result (success={previous_result.success}):\n"
                if previous_result.output:
                    result_message += f"Output: {previous_result.output}\n"
                if previous_result.error:
                    result_message += f"Error: {previous_result.error}\n"
                
                self.conversation_history.append({
                    "role": "user",
                    "content": result_message
                })
            
            # Get response from LLM with tools
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=self.conversation_history,
                tools=self.tools,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            assistant_message = response.choices[0].message
            
            # Add assistant response to conversation
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls
            })
            
            actions = []
            
            # Process tool calls
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    if tool_name in self.tool_mapping:
                        action = self.tool_mapping[tool_name](**tool_args)
                        actions.append(action)
                        
                        # Add tool result placeholder to conversation
                        # This will be filled in after action execution
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": "Action executed - result will be provided"
                        })
                    else:
                        self.logger.warning(f"Unknown tool: {tool_name}")
            
            # If no tool calls, treat as a thinking step
            if not actions:
                self.logger.info("No tool calls in response, treating as thinking step")
                # Return a placeholder action that will be skipped
                actions = [Action(
                    type=ActionType.COMMAND,
                    content="echo 'Thinking...'",
                    timeout=None,
                    working_directory=None
                )]
            
            self.logger.info(f"Generated {len(actions)} actions from tool calls")
            return actions
            
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {e}")
            return [Action(
                type=ActionType.DONE,
                content=f"Error: {str(e)}",
                timeout=None,
                working_directory=None
            )]
    
    async def add_context(self, context: str) -> None:
        """Add context information to the conversation"""
        self.conversation_history.append({
            "role": "user",
            "content": f"Context: {context}"
        })
    
    async def add_system_message(self, message: str) -> None:
        """Add a system message to the conversation"""
        self.conversation_history.append({
            "role": "system",
            "content": message
        })
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        return self.conversation_history.copy()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation"""
        user_messages = [msg for msg in self.conversation_history if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.conversation_history if msg["role"] == "assistant"]
        system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "system_messages": len(system_messages),
            "conversation_length": sum(len(msg["content"]) for msg in self.conversation_history)
        }
    
    async def reset_conversation(self) -> None:
        """Reset the conversation history"""
        self.conversation_history = []
        self.logger.info("Conversation history reset")
    
    async def save_conversation(self, filepath: str) -> None:
        """Save conversation history to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "config": self.config.model_dump(),
                    "conversation": self.conversation_history
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving conversation: {e}")
    
    async def load_conversation(self, filepath: str) -> bool:
        """Load conversation history from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.conversation_history = data.get("conversation", [])
                return True
        except Exception as e:
            self.logger.error(f"Error loading conversation: {e}")
            return False
    
    def truncate_conversation(self, max_messages: int = 20) -> None:
        """Truncate conversation history to keep recent messages"""
        if len(self.conversation_history) > max_messages:
            # Keep system messages and recent messages
            system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
            recent_messages = self.conversation_history[-(max_messages - len(system_messages)):]
            
            self.conversation_history = system_messages + recent_messages
            self.logger.info(f"Conversation truncated to {len(self.conversation_history)} messages")


class LLMAgentFactory:
    """Factory for creating LLM agents"""
    
    @staticmethod
    def create_agent(config: LLMConfig) -> LLMAgent:
        """Create an LLM agent with the given configuration"""
        return LLMAgent(config)
    
    @staticmethod
    def create_multiple_agents(config: LLMConfig, count: int) -> List[LLMAgent]:
        """Create multiple LLM agents with the same configuration"""
        return [LLMAgent(config) for _ in range(count)] 