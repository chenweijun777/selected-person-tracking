"""Public offline evaluation API. Requires the repository's frozen data assets."""

from ._run import evaluate_run

__all__ = ["evaluate_run"]
