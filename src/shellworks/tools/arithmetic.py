"""
directory: src/shellworks/tools/
file:      arithmetic.py

The one tool for Lesson 1.

This is ordinary deterministic Python code. It is executed only by the
orchestrator after validation succeeds. The model never calls this directly.
"""


def add_numbers(a: float, b: float) -> str:
    """Add two numbers and return the sum as a string."""
    result = a + b
    return str(result)
