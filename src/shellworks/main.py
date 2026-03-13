"""
directory: src/shellworks/
file:      main.py

Entry point for the shellworks Lesson 1 demo.

Usage:
    uv run shellworks


This file is intentionally thin. All the lesson logic lives in:
    shellworks/orchestrator/minimal_tool_calling.py
"""
import argparse
from dotenv import load_dotenv
from shellworks.app import create_app

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 1: Minimal tool calling in the terminal."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw request/response JSON to stderr for debugging.",
    )
    args = parser.parse_args()
    create_app(debug=args.debug).run()


if __name__ == "__main__":
    main()