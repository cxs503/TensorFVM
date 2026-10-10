"""Common, source-bound analytic CFD benchmark and publication report workflow."""
from .core import Metric, evaluate, relative_error
__all__ = ["Metric", "evaluate", "relative_error"]
