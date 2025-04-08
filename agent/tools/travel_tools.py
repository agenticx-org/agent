from agent.tools.registry import Tool, ToolRegistry


def create_travel_tools():
    """
    Create a set of travel-related tools.

    Returns:
        ToolRegistry: A registry with travel tools
    """
    registry = ToolRegistry()

    # Search web tool
    def search_web_handler(args):
        query = args.get("query", "").lower()
        if "weather" in query and "paris" in query:
            return {
                "result": (
                    "In late March, Paris typically has mild weather with average temperatures "
                    "ranging from 7°C to 14°C (45°F to 57°F). It can be somewhat unpredictable "
                    "with occasional rain showers. Spring is beginning, and you might see early "
                    "blooms in parks and gardens. It's advisable to bring layers and a light "
                    "waterproof jacket."
                )
            }
        else:
            return {
                "result": f"Search results for '{args.get('query')}': [mock search results]"
            }

    search_web_tool = Tool(
        name="search_web",
        description="Search the web for information on a topic",
        parameters={
            "query": {
                "type": "string",
                "description": "The search query to use",
                "required": True,
            }
        },
        handler=search_web_handler,
    )
    registry.register_tool(search_web_tool)

    # Get weather tool
    def get_weather_handler(args):
        location = args.get("location", "unknown")
        unit = args.get("unit", "celsius")
        if location.lower() == "paris":
            return {
                "result": (
                    f"Weather in Paris in late March: Average temperatures between 7-14°{unit[0].upper()}. "
                    f"Expect some rain showers and partly cloudy days. Spring is just beginning."
                )
            }
        else:
            return {"result": f"Weather in {location}: 22°{unit[0].upper()} and sunny"}

    get_weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "location": {
                "type": "string",
                "description": "The city or location to get weather for",
                "required": True,
            },
            "unit": {
                "type": "string",
                "description": "The unit of temperature ('celsius' or 'fahrenheit')",
                "required": False,
            },
        },
        handler=get_weather_handler,
    )
    registry.register_tool(get_weather_tool)

    # Airbnb search tool
    def airbnb_search_handler(args):
        location = args.get("location", "unknown")
        checkin = args.get("checkin", "")
        checkout = args.get("checkout", "")
        if location.lower() == "paris":
            return {
                "result": (
                    f"Found 15 properties in Paris for {checkin} to {checkout}:\n"
                    "1. Cozy apartment in Le Marais - €120/night - Superhost - 9.2 rating\n"
                    "2. Luxury studio near Eiffel Tower - €190/night - 8.9 rating\n"
                    "3. Charming flat in Montmartre - €105/night - Superhost - 9.5 rating\n"
                    "4. Modern loft in Latin Quarter - €150/night - 8.7 rating\n"
                    "5. Historic apartment near Louvre - €175/night - Superhost - 9.0 rating"
                )
            }
        else:
            return {
                "result": f"Found 15 properties in {location} for {checkin} to {checkout}:\n"
                + "1. Cozy apartment in city center - $120/night\n"
                + "2. Luxury condo with pool - $250/night\n"
                + "3. Charming studio near attractions - $95/night"
            }

    airbnb_search_tool = Tool(
        name="airbnb_search",
        description="Search for Airbnb listings in a location",
        parameters={
            "location": {
                "type": "string",
                "description": "The city or location to search in",
                "required": True,
            },
            "checkin": {
                "type": "string",
                "description": "The check-in date (YYYY-MM-DD)",
                "required": True,
            },
            "checkout": {
                "type": "string",
                "description": "The check-out date (YYYY-MM-DD)",
                "required": True,
            },
        },
        handler=airbnb_search_handler,
    )
    registry.register_tool(airbnb_search_tool)

    return registry
