import os
import sys

# Add parent directory to path to import from agent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from agent.tools.utility_tools import create_utility_tools


def main():
    """
    Example demonstrating how to create and use custom tools with the agent.
    """
    # Create a registry with utility tools
    tool_registry = create_utility_tools()

    # Create the agent with the tool registry
    agent = Agent(
        name="Utility Assistant",
        description="An agent that helps with various utility functions",
        system_prompt="You are a helpful assistant that can perform various utility functions like getting the current time, performing calculations, and transforming text.",
        tool_registry=tool_registry,
        max_steps=5,
    )

    # Example user query
    user_input = "I need help with the following: 1) What's the current date and time? 2) Calculate 15 * 24 + 7. 3) Convert 'hello world' to uppercase."

    print("\n==== Custom Tools Example ====")
    print(f"User Query: {user_input}\n")

    # Run the process loop
    result = agent.process_loop(user_input)

    # Print the result
    print("\n=== Final Result ===")
    if result["status"] == "success":
        print(f"Agent response after {result['turns']} turns:\n")
        if result["final_response"]:
            print(result["final_response"])
        else:
            print("No final response from agent")
    else:
        print(f"Error: {result['message']}")

    # Print conversation statistics
    print(f"\nConversation summary:")
    print(f"- Total turns: {result['turns']}")
    print(f"- Messages: {len(result['conversation'])}")

    # Count tool usage
    tool_usage = {}
    for message in result["conversation"]:
        if message["role"] == "function":
            tool_name = message["name"]
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

    if tool_usage:
        print("- Tool usage:")
        for tool_name, count in tool_usage.items():
            print(f"  - {tool_name}: {count} times")
    else:
        print("- No tools were used")


if __name__ == "__main__":
    main()
