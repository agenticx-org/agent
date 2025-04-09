#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

from agent.agent import Agent

# Load environment variables
load_dotenv()

# Configuration from env vars
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "claude-3-7-sonnet-latest")
RAW_AUTHORIZED_IMPORTS = os.getenv("AUTHORIZED_IMPORTS", "math,random,datetime,json,re")
AUTHORIZED_IMPORTS = [
    imp.strip() for imp in RAW_AUTHORIZED_IMPORTS.split(",") if imp.strip()
]


def configure_logging(verbose=False):
    """Configure logging settings."""
    if verbose:
        # Log to console if verbose mode is enabled
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        # Log to file only, keeping console clean for JSON output
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(log_dir, "agent.log")),
                # No stream handler to avoid console output
            ],
        )

    # Silence noisy third-party library logs regardless of verbose setting
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    """Main entry point for the CLI agent."""
    parser = argparse.ArgumentParser(
        description="Run Agentic Library from the command line."
    )
    parser.add_argument("task", help="The task for the agent to perform.")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging to console"
    )
    args = parser.parse_args()

    # Configure logging based on verbose flag
    configure_logging(verbose=args.verbose)

    logger = logging.getLogger("CLI_Agent")

    if not ANTHROPIC_API_KEY:
        print(
            json.dumps(
                {
                    "type": "error",
                    "content": "No Anthropic credentials found in environment variables. Please set ANTHROPIC_API_KEY.",
                }
            )
        )
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
        print(
            json.dumps(
                {"type": "error", "content": f"An unexpected error occurred: {e}"}
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
