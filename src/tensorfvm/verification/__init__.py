"""Common, source-bound analytic CFD benchmark and publication report workflow."""
from .core import Metric, evaluate, relative_error
from . import kmm1987

__all__ = ["Metric", "evaluate", "relative_error", "kmm1987"]
