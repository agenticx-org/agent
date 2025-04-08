import os
from textwrap import dedent

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.e2b import E2BTools
from agno.tools.thinking import ThinkingTools
from dotenv import load_dotenv

from prompts import agentic_loop, system_prompt

# Load environment variables from .env file
load_dotenv()

e2b_tools = E2BTools(
    timeout=600,
)

agent = Agent(
    model=Claude(id="claude-3-7-sonnet-latest", api_key=os.getenv("ANTHROPIC_API_KEY")),
    tools=[ThinkingTools(), e2b_tools],
    show_tool_calls=True,
    system_message=system_prompt,
    instructions=agentic_loop,
    markdown=True,
    reasoning_max_steps=50,
    debug_mode=True,
)

agent.print_response(
    "Write Python code to generate the first 10 Fibonacci numbers and calculate their sum and average"
)
