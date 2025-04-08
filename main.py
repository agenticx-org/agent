from agent import Agent
from agent.tools.thinking_tools import ThinkingTools


def main():
    # Create a thinking tools instance
    thinking_tools = ThinkingTools()

    # Create a sample agent with a specific model and thinking tools
    sample_agent = Agent(
        name="Research Assistant",
        description="An agent that helps with research tasks",
        system_prompt="""You are a helpful research assistant that provides accurate information.

When using the think tool, follow these guidelines:
1. Break down complex topics into distinct aspects or perspectives
2. Use a separate think call for each distinct aspect or perspective
3. Build upon previous thoughts to develop a comprehensive analysis
4. Never repeat the same thought multiple times
5. After analyzing from multiple angles, synthesize the insights into a final response

For example, when analyzing AI impact:
- First thought: Economic implications
- Second thought: Social and cultural changes
- Third thought: Safety and security considerations
- Fourth thought: Ethical implications
- Final response: Synthesis of all perspectives""",
        tool_registry=thinking_tools.get_registry(),
        max_steps=10,
        model_id="claude-3-7-sonnet-latest",
    )

    # Run the agent with some input that requires reasoning
    result = sample_agent.process_loop(
        "Explain the potential impact of large-scale agentic AI systems on society. Think through multiple perspectives."
    )
    print("\nAgent response:")
    print(result["final_response"] if "final_response" in result else result)

    # Reset the agent for a new task
    sample_agent.reset()


if __name__ == "__main__":
    main()
