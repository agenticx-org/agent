class Tool:
    """
    Represents a single tool that an agent can use.
    """

    def __init__(self, name, description, parameters=None, handler=None):
        """
        Initialize a Tool.

        Args:
            name (str): The name of the tool
            description (str): A description of what the tool does
            parameters (dict, optional): Parameters the tool accepts. Defaults to None.
            handler (callable, optional): A function that implements the tool's behavior. Defaults to None.
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.handler = handler

    def to_dict(self):
        """
        Convert the tool to a dictionary representation.

        Returns:
            dict: The tool as a dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def execute(self, args):
        """
        Execute the tool with the given arguments.

        Args:
            args (dict): Arguments to pass to the tool

        Returns:
            dict: The result of the tool execution
        """
        if self.handler is None:
            return {"error": f"No handler implemented for tool '{self.name}'"}

        try:
            return self.handler(args)
        except Exception as e:
            return {"error": f"Error executing tool '{self.name}': {str(e)}"}


class ToolRegistry:
    """
    A registry of tools that an agent can use.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools = {}

    def register_tool(self, tool):
        """
        Register a tool in the registry.

        Args:
            tool (Tool): The tool to register

        Returns:
            ToolRegistry: The registry (for method chaining)
        """
        self.tools[tool.name] = tool
        return self

    def get_tool(self, name):
        """
        Get a tool by name.

        Args:
            name (str): The name of the tool

        Returns:
            Tool: The tool, or None if not found
        """
        return self.tools.get(name)

    def list_tools(self):
        """
        Get a list of all registered tools.

        Returns:
            list: A list of all registered tools
        """
        return list(self.tools.values())

    def list_tool_dicts(self):
        """
        Get a list of all registered tools as dictionaries.

        Returns:
            list: A list of all registered tools as dictionaries
        """
        return [tool.to_dict() for tool in self.tools.values()]

    def execute_tool(self, name, args):
        """
        Execute a tool by name.

        Args:
            name (str): The name of the tool
            args (dict): Arguments to pass to the tool

        Returns:
            dict: The result of the tool execution
        """
        tool = self.get_tool(name)
        if tool is None:
            return {"error": f"Tool '{name}' not found in registry"}

        return tool.execute(args)
