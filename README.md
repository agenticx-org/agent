# Research Agent

A modular Claude 3.7 Sonnet-powered agent for conducting research and solving complex tasks step-by-step.

## Project Structure

The codebase has been refactored into a modular structure:

```
├── agent/                    # Agent package
│   ├── __init__.py           # Package initialization
│   ├── agent.py              # Main Agent class
│   ├── code_executor.py      # Python code execution
│   ├── llm.py                # LLM interaction
│   ├── prompts.py            # System prompt generation
│   ├── state_manager.py      # State management
│   └── tools.py              # Tool implementations and manager
├── .env                      # Environment variables (create from .env.example)
├── .env.example              # Example environment file
├── main.py                   # CLI entry point
└── README.md                 # This file
```

## Setup

1. Clone the repository
2. Create a `.env` file from `.env.example`
3. Add your Anthropic API key to the `.env` file

## Usage

Run the agent with a task description:

```bash
python main.py "Your research or analysis task here"
```

## Features

- Step-by-step reasoning and planning
- Python code execution with state persistence
- Task-specific searches (simulated)
- Markdown-based plan and findings tracking
- Detailed logging

## Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude access
- `MODEL_ID`: Claude model ID (defaults to `claude-3-7-sonnet-latest`)
- `AUTHORIZED_IMPORTS`: Comma-separated list of allowed Python imports

## Adding Tools

To add new tools:

1. Add tool implementation functions in `agent/tools.py`
2. Register tools in the `ToolManager._load_tools` method
