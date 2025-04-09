"""
This file has been refactored into a modular structure.

The code is now organized in the 'agent' directory with the following modules:
- agent/__init__.py - Package initialization
- agent/agent.py - Main Agent class
- agent/llm.py - LLM interaction
- agent/code_executor.py - Python code execution
- agent/state_manager.py - State management
- agent/tools.py - Tool implementations and ToolManager
- agent/prompts.py - System prompt generation

Please use the main.py file to run the agent.
"""

# For backward compatibility, import the refactored Agent
from agent.agent import Agent

if __name__ == "__main__":
    print("This file has been refactored. Please use main.py instead.")
    print("Run: python main.py 'your task here'")
