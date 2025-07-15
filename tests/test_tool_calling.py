#!/usr/bin/env python3
"""
Test script to demonstrate tool calling functionality with OpenRouter.
Requires OPENROUTER_API_KEY environment variable to be set.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import LLMConfig
from src.agents.llm_agent import LLMAgent

async def test_tool_calling():
    """Test tool calling with OpenRouter"""
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("Please set it with: export OPENROUTER_API_KEY=your_key_here")
        return False
    
    print("🧪 Testing Tool Calling with OpenRouter...")
    
    try:
        # Create LLM config
        config = LLMConfig(
            model="anthropic/claude-4-sonnet",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Create agent
        agent = LLMAgent(config)
        
        # Initialize conversation
        template_prompt = """You are a helpful assistant working in a test environment."""
        environment_prompt = """Please help me create a simple Python script that prints 'Hello, World!' to test your tool calling abilities."""
        
        await agent.initialize_conversation(template_prompt, environment_prompt)
        
        print("✓ Agent initialized")
        
        # Get first action
        actions = await agent.get_next_action()
        
        if actions:
            print(f"✓ Received {len(actions)} actions")
            for i, action in enumerate(actions, 1):
                print(f"  Action {i}: {action.type} - {action.content[:50]}...")
        else:
            print("❌ No actions received")
            return False
        
        # Simulate a successful execution result
        from src.core.config import ActionResult
        from datetime import datetime
        
        mock_result = ActionResult(
            success=True,
            output="Hello, World!",
            duration=0.5,
            timestamp=datetime.now().isoformat()
        )
        
        # Get next action with result
        actions = await agent.get_next_action(mock_result)
        
        if actions:
            print(f"✓ Received {len(actions)} follow-up actions")
            for i, action in enumerate(actions, 1):
                print(f"  Action {i}: {action.type} - {action.content[:50]}...")
        
        print("🎉 Tool calling test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tool_calling())
    exit(0 if success else 1) 