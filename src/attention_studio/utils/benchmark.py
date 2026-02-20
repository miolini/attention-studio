from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    std_dev: float
    memory_used_mb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Benchmark:
    def __init__(self, warmup_runs: int = 2, verbose: bool = True):
        self.warmup_runs = warmup_runs
        self.verbose = verbose

    def run(
        self,
        func: Callable,
        iterations: int = 10,
        *args,
        **kwargs,
    ) -> BenchmarkResult:
        for _ in range(self.warmup_runs):
            func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

        times = []
        memory_used = []
        mem_before = 0

        for _ in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()

            start = time.perf_counter()
            func(*args, **kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated()
                memory_used.append((mem_after - mem_before) / 1024**2)

            end_time = time.perf_counter()
            times.append(end_time - start)

        avg_time = sum(times) / len(times)
        std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

        result = BenchmarkResult(
            name=getattr(func, "__name__", "unknown"),
            iterations=iterations,
            total_time=sum(times),
            average_time=avg_time,
            min_time=min(times),
            max_time=max(times),
            std_dev=std_dev,
            memory_used_mb=max(memory_used) if memory_used else None,
        )

        if self.verbose:
            self._print_result(result)

        return result

    def _print_result(self, result: BenchmarkResult):
        print(f"\n{'='*50}")
        print(f"Benchmark: {result.name}")
        print(f"{'='*50}")
        print(f"Iterations: {result.iterations}")
        print(f"Total time: {result.total_time:.4f}s")
        print(f"Average time: {result.average_time:.4f}s")
        print(f"Min time: {result.min_time:.4f}s")
        print(f"Max time: {result.max_time:.4f}s")
        print(f"Std dev: {result.std_dev:.4f}s")
        if result.memory_used_mb:
            print(f"Memory used: {result.memory_used_mb:.2f} MB")
        print(f"{'='*50}\n")


class BenchmarkSuite:
    def __init__(self, name: str = "Benchmark Suite"):
        self.name = name
        self._results: dict[str, BenchmarkResult] = {}

    def add(self, name: str, result: BenchmarkResult):
        self._results[name] = result

    def get_result(self, name: str) -> BenchmarkResult | None:
        return self._results.get(name)

    def get_all_results(self) -> dict[str, BenchmarkResult]:
        return self._results

    def compare(self, metric: str = "average_time") -> list[tuple[str, float]]:
        items = []
        for name, result in self._results.items():
            if metric == "average_time":
                items.append((name, result.average_time))
            elif metric == "total_time":
                items.append((name, result.total_time))
            elif metric == "memory_used_mb":
                items.append((name, result.memory_used_mb or 0))
        return sorted(items, key=lambda x: x[1])

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"{self.name}")
        print(f"{'='*60}")

        if not self._results:
            print("No benchmarks run yet.")
            return

        fastest = self.compare("average_time")
        if fastest:
            print(f"\nFastest: {fastest[0][0]} ({fastest[0][1]:.4f}s)")

        for name, result in sorted(self._results.items(), key=lambda x: x[1].average_time):
            print(f"\n{name}:")
            print(f"  Avg: {result.average_time:.4f}s")
            print(f"  Total: {result.total_time:.4f}s")
            if result.memory_used_mb:
                print(f"  Memory: {result.memory_used_mb:.2f} MB")

        print(f"\n{'='*60}\n")


def benchmark_function(func: Callable, iterations: int = 10, warmup: int = 2) -> BenchmarkResult:
    bench = Benchmark(warmup_runs=warmup, verbose=False)
    return bench.run(func, iterations)


def compare_functions(
    funcs: list[tuple[str, Callable]],
    iterations: int = 10,
) -> BenchmarkSuite:
    suite = BenchmarkSuite()
    bench = Benchmark(warmup_runs=2, verbose=False)

    for name, func in funcs:
        result = bench.run(func, iterations)
        suite.add(name, result)

    return suite
