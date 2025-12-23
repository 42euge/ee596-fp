"""Registry for evaluation benchmarks."""

from typing import Dict, List, Optional, Type

from .benchmarks.base import BaseBenchmark, EvaluationResult


class BenchmarkRegistry:
    """Registry for managing evaluation benchmarks."""

    _benchmarks: Dict[str, Type[BaseBenchmark]] = {}

    @classmethod
    def register(cls, name: str, benchmark_class: Type[BaseBenchmark]):
        """Register a benchmark.

        Args:
            name: Benchmark name
            benchmark_class: Benchmark class
        """
        cls._benchmarks[name] = benchmark_class
        print(f"Registered benchmark: {name}")

    @classmethod
    def get_benchmark(cls, name: str) -> Optional[Type[BaseBenchmark]]:
        """Get a benchmark class by name.

        Args:
            name: Benchmark name

        Returns:
            Benchmark class, or None if not found
        """
        return cls._benchmarks.get(name)

    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all registered benchmarks.

        Returns:
            List of benchmark names
        """
        return list(cls._benchmarks.keys())

    @classmethod
    def evaluate(
        cls,
        model,
        benchmark_name: str,
        split: str = "test",
        num_samples: Optional[int] = None,
        **generation_kwargs
    ) -> EvaluationResult:
        """Evaluate model on a benchmark.

        Args:
            model: Model to evaluate
            benchmark_name: Name of benchmark
            split: Dataset split
            num_samples: Number of samples to evaluate
            **generation_kwargs: Generation arguments

        Returns:
            EvaluationResult object

        Raises:
            ValueError: If benchmark not found
        """
        benchmark_class = cls.get_benchmark(benchmark_name)
        if benchmark_class is None:
            raise ValueError(
                f"Benchmark '{benchmark_name}' not found. "
                f"Available benchmarks: {cls.list_benchmarks()}"
            )

        benchmark = benchmark_class()
        return benchmark.evaluate(model, split, num_samples, **generation_kwargs)

    @classmethod
    def evaluate_all(
        cls,
        model,
        benchmark_names: Optional[List[str]] = None,
        split: str = "test",
        num_samples: Optional[int] = None,
        **generation_kwargs
    ) -> Dict[str, EvaluationResult]:
        """Evaluate model on multiple benchmarks.

        Args:
            model: Model to evaluate
            benchmark_names: List of benchmark names (None = all)
            split: Dataset split
            num_samples: Number of samples per benchmark
            **generation_kwargs: Generation arguments

        Returns:
            Dictionary mapping benchmark names to results
        """
        if benchmark_names is None:
            benchmark_names = cls.list_benchmarks()

        results = {}
        for benchmark_name in benchmark_names:
            print(f"\nEvaluating on {benchmark_name}...")
            try:
                result = cls.evaluate(
                    model, benchmark_name, split, num_samples, **generation_kwargs
                )
                results[benchmark_name] = result
                print(f"  Accuracy: {result.metrics['accuracy']:.3f}")
            except Exception as e:
                print(f"  Error: {e}")

        return results
