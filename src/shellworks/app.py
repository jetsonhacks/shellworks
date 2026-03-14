"""
directory: src/shellworks/
file:      app.py

ShellworksApp — the interactive shell loop.

------------------
build_provider() is called once at startup. The resulting provider is
passed into run_turn each turn. This separates infrastructure setup
from the per-turn orchestration loop.

The active endpoint is controlled by the SHELLWORKS_ENDPOINT environment
variable. Set it to the name of a file in configs/endpoints/ (without
the .toml extension):

  SHELLWORKS_ENDPOINT=local_thor   uv run shellworks
  SHELLWORKS_ENDPOINT=local_orin   uv run shellworks

If unset, the name "local_primary" is used, which resolves to
configs/endpoints/local_primary.toml. That file should point at
whichever server is running locally.

The active model defaults to the endpoint's default_model. Override it
with SHELLWORKS_MODEL for a single run without changing the saved config.
"""

import sys
import readline  # noqa: F401 — imported for side effects (enables arrow keys, history)


from shellworks.llm.provider import build_provider
from shellworks.orchestrator.minimal_tool_calling import run_turn


class ShellworksApp:
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    def run(self) -> None:
        print("Shellworks — Agentic AI from First Principles")
        print("Type your request and press Enter. Type /help for available commands.\n")

        try:
            provider = build_provider()
        except (FileNotFoundError, ValueError) as exc:
            print(f"[startup error] {exc}", file=sys.stderr)
            sys.exit(1)

        if self.debug:
            print(
                f"[debug] Loaded endpoint:   {provider.endpoint.name}",
                file=sys.stderr,
            )
            print(
                f"[debug] Active model:      {provider.model}",
                file=sys.stderr,
            )
            print(
                f"[debug] Reasoning control: {provider.profile.reasoning_control}",
                file=sys.stderr,
            )
            print()

        reasoning = False  # session default

        while True:
            prompt = "shellworks(reasoning)> " if reasoning else "shellworks> "
            try:
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.", file=sys.stderr)
                break

            if not user_input:
                continue

            if user_input == "/reasoning on":
                reasoning = True
                print("[reasoning ON]")
            elif user_input == "/reasoning off":
                reasoning = False
                print("[reasoning OFF]")
            elif user_input == "/status":
                print(f"[reasoning {'ON' if reasoning else 'OFF'}]")
            elif user_input == "/exit":
                print("Exiting.")
                break
            elif user_input == "/help":
                print("  /reasoning on   enable reasoning mode")
                print("  /reasoning off  disable reasoning mode")
                print("  /status         show current reasoning mode")
                print("  /exit           exit shellworks")
            elif user_input.startswith("/"):
                print(f"Unknown command '{user_input}'. Type /help for available commands.")
            else:
                run_turn(user_input, provider=provider, reasoning=reasoning, debug=self.debug)
                print()


def create_app(debug: bool = False) -> ShellworksApp:
    return ShellworksApp(debug=debug)