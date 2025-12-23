"""Benchmark implementations."""

# Import benchmarks to auto-register them
from .gsm8k import GSM8KBenchmark

# Future benchmarks can be added here
# from .math import MATHBenchmark
# from .openrubrics import OpenRubricsBenchmark

__all__ = [
    "GSM8KBenchmark",
]
