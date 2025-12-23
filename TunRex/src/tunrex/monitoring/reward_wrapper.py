"""
Reward Function Wrapper for Automatic Monitoring

Wraps reward functions to automatically track calls, values, errors, and execution time.
"""

import time
import functools
from typing import Callable, Any, Dict, Optional
from collections import defaultdict
import warnings


class RewardFunctionWrapper:
    """
    Wrapper around a reward function that tracks its behavior.

    Monitors:
    - Number of calls
    - Execution time
    - Return values
    - Errors/exceptions
    - Input/output patterns
    """

    def __init__(
        self,
        reward_fn: Callable,
        name: Optional[str] = None,
        log_errors: bool = True,
        track_timing: bool = True,
    ):
        """
        Initialize reward function wrapper.

        Args:
            reward_fn: The reward function to wrap
            name: Optional name (defaults to function name)
            log_errors: Whether to log errors
            track_timing: Whether to track execution time
        """
        self.reward_fn = reward_fn
        self.name = name or getattr(reward_fn, "__name__", "unknown")
        self.log_errors = log_errors
        self.track_timing = track_timing

        # Tracking statistics
        self.call_count = 0
        self.error_count = 0
        self.total_time = 0.0
        self.min_time = float("inf")
        self.max_time = 0.0

        # Store recent values for debugging
        self.recent_values = []
        self.recent_errors = []
        self.max_recent = 100

        functools.update_wrapper(self, reward_fn)

    def __call__(self, *args, **kwargs) -> Any:
        """Call the wrapped reward function with monitoring."""
        self.call_count += 1
        start_time = time.time() if self.track_timing else None

        try:
            result = self.reward_fn(*args, **kwargs)

            # Track return value
            self.recent_values.append(result)
            if len(self.recent_values) > self.max_recent:
                self.recent_values.pop(0)

            return result

        except Exception as e:
            self.error_count += 1

            # Track error
            error_info = {
                "error": str(e),
                "type": type(e).__name__,
                "args": args,
                "kwargs": kwargs,
            }
            self.recent_errors.append(error_info)
            if len(self.recent_errors) > self.max_recent:
                self.recent_errors.pop(0)

            if self.log_errors:
                warnings.warn(
                    f"Error in reward function '{self.name}': {e}\n"
                    f"Args: {args}, Kwargs: {kwargs}"
                )

            raise

        finally:
            if self.track_timing and start_time is not None:
                elapsed = time.time() - start_time
                self.total_time += elapsed
                self.min_time = min(self.min_time, elapsed)
                self.max_time = max(self.max_time, elapsed)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this reward function."""
        avg_time = self.total_time / self.call_count if self.call_count > 0 else 0.0

        return {
            "name": self.name,
            "calls": self.call_count,
            "errors": self.error_count,
            "error_rate": self.error_count / self.call_count if self.call_count > 0 else 0.0,
            "total_time": self.total_time,
            "avg_time": avg_time,
            "min_time": self.min_time if self.min_time != float("inf") else 0.0,
            "max_time": self.max_time,
            "recent_values": self.recent_values[-10:],  # Last 10 values
            "recent_errors": self.recent_errors[-5:],  # Last 5 errors
        }

    def reset(self):
        """Reset all tracking statistics."""
        self.call_count = 0
        self.error_count = 0
        self.total_time = 0.0
        self.min_time = float("inf")
        self.max_time = 0.0
        self.recent_values = []
        self.recent_errors = []


class RewardFunctionMonitor:
    """
    Manages multiple wrapped reward functions and aggregates their statistics.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize reward function monitor.

        Args:
            verbose: Whether to print monitoring information
        """
        self.wrapped_functions: Dict[str, RewardFunctionWrapper] = {}
        self.verbose = verbose

    def wrap(
        self,
        reward_fn: Callable,
        name: Optional[str] = None,
        log_errors: bool = True,
        track_timing: bool = True,
    ) -> RewardFunctionWrapper:
        """
        Wrap a reward function for monitoring.

        Args:
            reward_fn: The reward function to wrap
            name: Optional name
            log_errors: Whether to log errors
            track_timing: Whether to track timing

        Returns:
            Wrapped reward function
        """
        wrapper = RewardFunctionWrapper(
            reward_fn=reward_fn,
            name=name,
            log_errors=log_errors,
            track_timing=track_timing,
        )

        self.wrapped_functions[wrapper.name] = wrapper

        if self.verbose:
            print(f"✓ Wrapped reward function: {wrapper.name}")

        return wrapper

    def wrap_all(
        self,
        reward_fns: list[Callable],
        **kwargs,
    ) -> list[RewardFunctionWrapper]:
        """
        Wrap multiple reward functions.

        Args:
            reward_fns: List of reward functions to wrap
            **kwargs: Arguments to pass to wrap()

        Returns:
            List of wrapped reward functions
        """
        return [self.wrap(fn, **kwargs) for fn in reward_fns]

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all wrapped reward functions."""
        return {
            name: wrapper.get_stats()
            for name, wrapper in self.wrapped_functions.items()
        }

    def get_metrics_dict(self) -> Dict[str, float]:
        """
        Get metrics as flat dictionary for logging.

        Returns:
            Dictionary with execution metrics for all reward functions
        """
        metrics = {}

        for name, wrapper in self.wrapped_functions.items():
            stats = wrapper.get_stats()

            metrics[f"reward_fn/{name}/calls"] = stats["calls"]
            metrics[f"reward_fn/{name}/errors"] = stats["errors"]
            metrics[f"reward_fn/{name}/error_rate"] = stats["error_rate"]
            metrics[f"reward_fn/{name}/avg_time_ms"] = stats["avg_time"] * 1000
            metrics[f"reward_fn/{name}/max_time_ms"] = stats["max_time"] * 1000

        return metrics

    def print_summary(self):
        """Print a summary of all reward function statistics."""
        print("\n" + "=" * 80)
        print("Reward Function Execution Summary")
        print("=" * 80)

        for name, wrapper in self.wrapped_functions.items():
            stats = wrapper.get_stats()

            print(f"\n{name}:")
            print(f"  Calls: {stats['calls']}")
            print(f"  Errors: {stats['errors']} ({stats['error_rate']*100:.2f}%)")
            print(f"  Avg time: {stats['avg_time']*1000:.2f}ms")
            print(f"  Time range: [{stats['min_time']*1000:.2f}ms, {stats['max_time']*1000:.2f}ms]")

            if stats["recent_values"]:
                print(f"  Recent values: {stats['recent_values'][-5:]}")

            if stats["recent_errors"]:
                print(f"  Recent errors: {len(stats['recent_errors'])}")

        print("=" * 80 + "\n")

    def reset_all(self):
        """Reset statistics for all wrapped functions."""
        for wrapper in self.wrapped_functions.values():
            wrapper.reset()
