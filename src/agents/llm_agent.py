import openai
import json
import os
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import time

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
        self.pending_tool_calls: List[Dict[str, Any]] = []  # Store tool calls awaiting results
        self.logger = logging.getLogger("LLMAgent")
        
        # Token caching
        self.response_cache: Dict[str, Dict[str, Any]] = {}  # hash -> cached response
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Token usage tracking
        self.token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "api_calls": 0
        }
        
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
                                "description": "Optional timeout in seconds"
                            },
                            "working_directory": {
                                "type": "string",
                                "description": "Optional working directory"
                            },
                            "background": {
                                "type": "boolean",
                                "description": "Run command in background and don't wait for it"
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
            working_directory=kwargs.get("working_directory"),
            background=kwargs.get("background", False)
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

You are working in a Docker environment with shell access. You can execute commands, read/write files, and complete coding tasks. The unit tests will evaluate your work, but you don't have access to the test code itself.

You have access to the following tools:
1. execute_command - Run shell commands
2. write_file - Write content to files
3. read_file - Read content from files
4. mark_done - Mark the task as completed

Always think step by step and be methodical in your approach. When you're done with the task, use the mark_done tool to indicate completion.
"""
        
        self.conversation_history = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": environment_prompt.strip()}
        ]
        
        self.logger.info("Conversation initialized with system prompt and user message")
    
    async def get_next_action(self, previous_result: Optional[ActionResult] = None) -> Tuple[List[Action], List[Dict[str, Any]]]:
        """Get next action(s) from the LLM using tool calling with caching and truncation"""
        try:
            self.logger.debug(f"get_next_action called with previous_result: {previous_result is not None}")
            
            # Add tool results for previous actions if available
            if previous_result and self.pending_tool_calls:
                self.logger.debug(f"Adding tool results for {len(self.pending_tool_calls)} pending tool calls")
                # Add tool results to conversation
                for tool_call_info in self.pending_tool_calls:
                    result_content = f"Success: {previous_result.success}\n"
                    if previous_result.output:
                        # Truncate large outputs
                        truncated_output = self._truncate_output(previous_result.output)
                        result_content += f"Output: {truncated_output}\n"
                    if previous_result.error:
                        # Truncate large errors
                        truncated_error = self._truncate_output(previous_result.error)
                        result_content += f"Error: {truncated_error}\n"
                    if hasattr(previous_result, 'exit_code') and previous_result.exit_code is not None:
                        result_content += f"Exit code: {previous_result.exit_code}\n"
                    
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_info["id"],
                        "name": tool_call_info["name"],
                        "content": result_content.strip()
                    })
                    self.logger.debug(f"Added tool result for {tool_call_info['name']}: {result_content[:100]}...")
                
                # Clear pending tool calls
                self.pending_tool_calls = []
            
            # Truncate conversation if it's getting too long
            self.truncate_conversation()
            
            self.logger.debug(f"Conversation has {len(self.conversation_history)} messages")
            
            # Check cache if enabled
            conversation_hash = None
            if self.config.enable_caching:
                conversation_hash = self._get_conversation_hash(self.conversation_history)
                
                # Check if we have a cached response
                if conversation_hash in self.response_cache:
                    cached_response = self.response_cache[conversation_hash]
                    self.cache_hits += 1
                    self.logger.debug(f"Cache hit! Using cached response (hash: {conversation_hash[:8]})")
                    
                    # Return cached actions and tool calls
                    return cached_response["actions"], cached_response["tool_calls_info"]
                
                self.cache_misses += 1
                self.logger.debug(f"Cache miss, making API call (hash: {conversation_hash[:8]})")
            
            # Estimate input tokens for token usage tracking
            input_text = json.dumps(self.conversation_history)
            estimated_input_tokens = self._estimate_tokens(input_text)
            
            # Get response from LLM with tools
            self.logger.debug("Making API call to LLM...")
            start_time = time.time()
            
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=self.conversation_history,  # type: ignore
                tools=self.tools,  # type: ignore
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            api_duration = time.time() - start_time
            self.logger.debug(f"API call completed in {api_duration:.2f}s")
            
            if not response or not response.choices:
                self.logger.error("No response received from LLM")
                return [], []
            
            assistant_message = response.choices[0].message  # type: ignore
            content_preview = (assistant_message.content or "")[:100]
            self.logger.debug(f"Assistant response: {content_preview}...")
            
            # Update token usage tracking
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                self._update_token_usage(input_tokens, output_tokens)
                self.logger.debug(f"Token usage: {input_tokens} input, {output_tokens} output")
            else:
                # Fallback to estimation
                output_text = assistant_message.content or ""
                estimated_output_tokens = self._estimate_tokens(output_text)
                self._update_token_usage(estimated_input_tokens, estimated_output_tokens)
                self.logger.debug(f"Estimated token usage: {estimated_input_tokens} input, {estimated_output_tokens} output")
            
            # Convert tool_calls to proper dictionary format
            tool_calls_dict = None
            if assistant_message.tool_calls:
                self.logger.debug(f"Assistant made {len(assistant_message.tool_calls)} tool calls")
                tool_calls_dict = []
                for tool_call in assistant_message.tool_calls:
                    tool_calls_dict.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
                    self.logger.debug(f"Tool call: {tool_call.function.name} with args: {tool_call.function.arguments}")
            else:
                self.logger.debug("Assistant made no tool calls")
            
            # Add assistant response to conversation
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": tool_calls_dict
            })
            
            actions = []
            tool_calls_info = []
            
            # Process tool calls
            if assistant_message.tool_calls:
                self.logger.debug(f"Processing {len(assistant_message.tool_calls)} tool calls...")
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    self.logger.debug(f"Processing tool call: {tool_name}")
                    
                    if tool_name in self.tool_mapping:
                        action = self.tool_mapping[tool_name](**tool_args)
                        actions.append(action)
                        
                        # Store tool call info for later use
                        tool_call_info = {
                            "id": tool_call.id,
                            "name": tool_name,
                            "args": tool_args
                        }
                        tool_calls_info.append(tool_call_info)
                        
                        self.logger.debug(f"Created action: {action.type} - {action.content[:100]}...")
                    else:
                        self.logger.warning(f"Unknown tool: {tool_name}")
            
            # Store pending tool calls
            self.pending_tool_calls = tool_calls_info
            
            # Cache the response if caching is enabled
            if self.config.enable_caching and conversation_hash:
                # Clean up cache if it's getting too large
                if len(self.response_cache) >= self.config.cache_size:
                    # Remove oldest entries (simple FIFO)
                    oldest_key = next(iter(self.response_cache))
                    del self.response_cache[oldest_key]
                
                self.response_cache[conversation_hash] = {
                    "actions": actions,
                    "tool_calls_info": tool_calls_info,
                    "timestamp": time.time()
                }
                self.logger.debug(f"Cached response (hash: {conversation_hash[:8]})")
            
            # If no tool calls, return empty actions (task completion)
            if not actions:
                self.logger.debug("No tool calls in response, treating as task completion")
            
            self.logger.info(f"Generated {len(actions)} actions from tool calls")
            return actions, tool_calls_info
            
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {e}")
            return [], []
    
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
        self.pending_tool_calls = []
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
    
    def truncate_conversation(self, max_messages: Optional[int] = None) -> None:
        """Truncate conversation history to prevent context overflow while preserving task context"""
        max_messages = max_messages or self.config.max_context_messages
        
        if len(self.conversation_history) <= max_messages:
            return  # No truncation needed
        
        original_length = len(self.conversation_history)
        
        # Always preserve critical messages:
        # 1. System message (index 0) - contains instructions and tools
        # 2. Original task message (index 1) - contains the environment prompt/goal
        critical_messages = []
        
        if len(self.conversation_history) >= 1:
            critical_messages.append(self.conversation_history[0])  # System message
        
        if len(self.conversation_history) >= 2:
            critical_messages.append(self.conversation_history[1])  # Original task
        
        # Calculate how many recent messages we can keep
        available_slots = max_messages - len(critical_messages) - 1  # -1 for truncation notice
        
        if available_slots <= 0:
            # If we can't fit any recent messages, at least keep the critical ones
            recent_messages = []
            available_slots = 0
        else:
            # Keep the most recent messages
            recent_messages = self.conversation_history[-available_slots:]
        
        # Create truncation notice to inform the LLM
        truncation_notice = {
            "role": "system",
            "content": f"[CONTEXT TRUNCATED] The conversation history has been truncated to manage context size. "
                      f"Original conversation had {original_length} messages, now showing {len(critical_messages) + len(recent_messages) + 1} messages. "
                      f"Your original task and recent {len(recent_messages)} messages are preserved. "
                      f"Continue working on your assigned task."
        }
        
        # Reconstruct conversation: critical messages + truncation notice + recent messages
        self.conversation_history = critical_messages + [truncation_notice] + recent_messages
        
        self.logger.info(f"Conversation truncated from {original_length} to {len(self.conversation_history)} messages "
                        f"(preserved {len(critical_messages)} critical + {len(recent_messages)} recent + 1 truncation notice)")
    
    def _truncate_output(self, output: str) -> str:
        """Truncate long output to prevent context bloat"""
        if len(output) > self.config.max_output_length:
            truncated = output[:self.config.max_output_length]
            truncated += f"\n... [truncated {len(output) - self.config.max_output_length} characters]"
            return truncated
        return output
    
    def _get_conversation_hash(self, messages: List[Dict[str, Any]]) -> str:
        """Generate hash for conversation to enable caching"""
        # Create a stable representation of the conversation
        conversation_str = json.dumps(messages, sort_keys=True)
        return hashlib.md5(conversation_str.encode()).hexdigest()
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (approximately 4 chars per token)"""
        return len(text) // 4
    
    def _update_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Update token usage statistics"""
        if not self.config.track_token_usage:
            return
        
        self.token_usage["total_input_tokens"] += input_tokens
        self.token_usage["total_output_tokens"] += output_tokens
        self.token_usage["api_calls"] += 1
        
        # Warn if usage is high
        if self.config.warn_high_usage:
            total_tokens = input_tokens + output_tokens
            if total_tokens > 10000:  # 10K token warning threshold
                self.logger.warning(f"High token usage: {total_tokens:,} tokens")
    
    def get_token_usage_stats(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        return {
            **self.token_usage,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }

    async def add_tool_result(self, tool_call_id: str, tool_name: str, result: ActionResult) -> None:
        """Add tool result to conversation history"""
        # Format the result content
        result_content = f"Success: {result.success}\n"
        if result.output:
            result_content += f"Output: {result.output}\n"
        if result.error:
            result_content += f"Error: {result.error}\n"
        if hasattr(result, 'exit_code') and result.exit_code is not None:
            result_content += f"Exit code: {result.exit_code}\n"
        
        # Add tool result to conversation
        self.conversation_history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result_content.strip()
        })
        
        self.logger.debug(f"Added tool result for {tool_name}: {result.success}")


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