from datetime import datetime

from agent.tools.registry import Tool, ToolRegistry


def create_utility_tools():
    """
    Create a registry with utility tools.

    Returns:
        ToolRegistry: A registry containing utility tools
    """
    registry = ToolRegistry()

    # Current time tool
    def get_current_time_handler(args):
        """Get the current date and time"""
        format_str = args.get("format", "%Y-%m-%d %H:%M:%S")
        timezone = args.get("timezone", "UTC")

        try:
            now = datetime.now()
            formatted_time = now.strftime(format_str)
            return {"result": f"Current time ({timezone}): {formatted_time}"}
        except Exception as e:
            return {"error": f"Error formatting time: {str(e)}"}

    time_tool = Tool(
        name="get_current_time",
        description="Get the current date and time",
        parameters={
            "format": {
                "type": "string",
                "description": "Format string for the datetime (e.g. %Y-%m-%d %H:%M:%S)",
                "required": False,
            },
            "timezone": {
                "type": "string",
                "description": "Timezone (e.g. UTC, EST)",
                "required": False,
            },
        },
        handler=get_current_time_handler,
    )
    registry.register_tool(time_tool)

    # Calculator tool
    def calculator_handler(args):
        """Perform a simple calculation"""
        expression = args.get("expression", "")
        if not expression:
            return {"error": "No expression provided"}

        try:
            # SECURITY NOTE: In a real application, you would need to
            # sanitize/restrict the expression to prevent code execution
            # This is just a simple example
            result = eval(expression)
            return {"result": f"Result: {expression} = {result}"}
        except Exception as e:
            return {"error": f"Error calculating expression: {str(e)}"}

    calculator_tool = Tool(
        name="calculator",
        description="Perform a simple calculation",
        parameters={
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate (e.g. 2 + 2)",
                "required": True,
            }
        },
        handler=calculator_handler,
    )
    registry.register_tool(calculator_tool)

    # Text transformation tool
    def transform_text_handler(args):
        """Transform text based on the specified operation"""
        text = args.get("text", "")
        operation = args.get("operation", "uppercase")

        if not text:
            return {"error": "No text provided"}

        try:
            if operation.lower() == "uppercase":
                result = text.upper()
            elif operation.lower() == "lowercase":
                result = text.lower()
            elif operation.lower() == "capitalize":
                result = text.capitalize()
            elif operation.lower() == "reverse":
                result = text[::-1]
            else:
                return {"error": f"Unknown operation: {operation}"}

            return {"result": f"Transformed text ({operation}): {result}"}
        except Exception as e:
            return {"error": f"Error transforming text: {str(e)}"}

    text_tool = Tool(
        name="transform_text",
        description="Transform text using various operations",
        parameters={
            "text": {
                "type": "string",
                "description": "The text to transform",
                "required": True,
            },
            "operation": {
                "type": "string",
                "description": "The operation to perform (uppercase, lowercase, capitalize, reverse)",
                "required": False,
            },
        },
        handler=transform_text_handler,
    )
    registry.register_tool(text_tool)

    return registry
