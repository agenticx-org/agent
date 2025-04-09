#!/usr/bin/env python3
import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from agent.agent import Agent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CLI_Agent")

# Configuration from env vars
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "claude-3-7-sonnet-latest")
RAW_AUTHORIZED_IMPORTS = os.getenv("AUTHORIZED_IMPORTS", "math,random,datetime,json,re")
AUTHORIZED_IMPORTS = [
    imp.strip() for imp in RAW_AUTHORIZED_IMPORTS.split(",") if imp.strip()
]


def main():
    """Main entry point for the CLI agent."""
    parser = argparse.ArgumentParser(
        description="Run Agentic Library from the command line."
    )
    parser.add_argument("task", help="The task for the agent to perform.")
    args = parser.parse_args()

    print("*" * 50)
    print("         Command Line Agent Runner")
    print("*" * 50)

    if not ANTHROPIC_API_KEY:
        print("[ERROR] No Anthropic credentials found in environment variables.")
        print("Please set ANTHROPIC_API_KEY.")
        sys.exit(1)

    try:
        agent = Agent(
            task=args.task, authorized_imports=AUTHORIZED_IMPORTS, model_id=MODEL_ID
        )
        agent.run()
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during agent execution: {e}", exc_info=True
        )
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
