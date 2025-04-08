from textwrap import dedent
from typing import Optional

from agent.tools.registry import Tool, ToolRegistry


class ThinkingTools:
    """
    Thinking tools for agents to use as a scratchpad or for complex reasoning.
    """

    def __init__(
        self,
        think: bool = True,
        instructions: Optional[str] = None,
        add_instructions: bool = False,
    ):
        """
        Initialize ThinkingTools.

        Args:
            think (bool, optional): Whether to include the think tool. Defaults to True.
            instructions (str, optional): Custom instructions for the tools. Defaults to None.
            add_instructions (bool, optional): Whether to add instructions to the system prompt. Defaults to False.
        """
        self.registry = ToolRegistry()
        self.instructions = instructions
        self.add_instructions = add_instructions

        if instructions is None:
            self.instructions = dedent(
                """\
            ## Using the think tool
            Before taking any action or responding to the user after receiving tool results, use the think tool as a scratchpad to:
            - List the specific rules that apply to the current request
            - Check if all required information is collected
            - Verify that the planned action complies with all policies
            - Iterate over tool results for correctness

            ## Rules
            - Use the think tool generously to jot down thoughts and ideas.\
            """
            )

        if think:
            # Create the think tool
            think_tool = Tool(
                name="think",
                description="Use this tool to think through a problem or explain your reasoning. It will not obtain new information or take any actions, but just serve as a scratchpad.",
                parameters={
                    "thought": {
                        "type": "string",
                        "description": "A thought to think about and analyze",
                        "required": True,
                    }
                },
                handler=self.think,
            )
            self.registry.register_tool(think_tool)

    def register(self, handler):
        """
        Register a custom handler function as a tool.

        Args:
            handler: The handler function to register
        """
        # Extract function name and create tool from it
        name = handler.__name__
        description = handler.__doc__ or f"Use the {name} tool"

        # Create a simple tool with the handler
        tool = Tool(
            name=name,
            description=description,
            parameters={
                "thought": {
                    "type": "string",
                    "description": "A thought to process",
                    "required": True,
                }
            },
            handler=handler,
        )

        self.registry.register_tool(tool)

    def think(self, agent, thought: str) -> str:
        """Use the tool to think about something.
        It will not obtain new information or take any actions, but just append the thought to the log and return the result.
        Use it when complex reasoning or some cache memory or a scratchpad is needed.

        Args:
            agent: The agent instance
            thought: A thought to think about and log

        Returns:
            str: The full log of thoughts and the new thought
        """
        try:
            # Return just the current thought since we can't persist state
            formatted_thought = dedent(
                f"""Thought:
                - {thought}
                """
            ).strip()
            return formatted_thought
        except Exception as e:
            return f"Error recording thought: {e}"

    def get_registry(self):
        """
        Get the tool registry.

        Returns:
            ToolRegistry: The registry with thinking tools
        """
        return self.registry
