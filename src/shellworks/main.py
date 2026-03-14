"""
directory: src/shellworks/
file:      main.py

Entry point for the shellworks demo.

Usage:
    uv run shellworks
    uv run shellworks --debug

Environment variables (keep these in a .env file):
    SHELLWORKS_ENDPOINT     Endpoint config name, without .toml extension.
                            Resolves to configs/endpoints/<name>.toml.
                            Default: local_primary

    SHELLWORKS_MODEL        Override the active model for this run.
                            Must match canonical_model in a model profile.
                            Default: endpoint's default_model

    SHELLWORKS_CONFIGS_DIR  Path to the configs directory.
                            Default: ./configs

This file is intentionally thin. All lesson logic lives in:
    shellworks/orchestrator/minimal_tool_calling.py
"""

import argparse

from dotenv import load_dotenv

from shellworks.app import create_app

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal tool calling with endpoints and model profiles."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print request/response detail to stderr.",
    )
    args = parser.parse_args()
    create_app(debug=args.debug).run()


if __name__ == "__main__":
    main()