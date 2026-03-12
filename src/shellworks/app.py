import sys
from shellworks.orchestrator.minimal_tool_calling import run_turn


class ShellworksApp:
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    def run(self) -> None:
        print("shellworks — Lesson 1: Tool Calling from First Principles")
        print("Type your request and press Enter. Press Ctrl-C or Ctrl-D to quit.\n")

        while True:
            try:
                user_input = input("shellworks> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.", file=sys.stderr)
                break

            if not user_input:
                continue

            run_turn(user_input, debug=self.debug)
            print()


def create_app(debug: bool = False) -> ShellworksApp:
    return ShellworksApp(debug=debug)