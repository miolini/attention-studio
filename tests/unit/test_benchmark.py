import pytest
import time
import torch
import torch.nn as nn

from attention_studio.utils.benchmark import (
    Benchmark,
    BenchmarkSuite,
    benchmark_function,
    compare_functions,
    BenchmarkResult,
)


class TestBenchmark:
    def test_basic_benchmark(self):
        def simple_func():
            x = torch.randn(100, 100)
            _ = x * 2

        bench = Benchmark(warmup_runs=0, verbose=False)
        result = bench.run(simple_func, iterations=3)

        assert isinstance(result, BenchmarkResult)
        assert result.name == "simple_func"
        assert result.iterations == 3
        assert result.average_time > 0

    def test_benchmark_with_args(self):
        def func_with_args():
            a = 1
            b = 2
            return a + b

        bench = Benchmark(warmup_runs=0, verbose=False)
        result = bench.run(func_with_args, iterations=2)

        assert result.average_time >= 0

    def test_benchmark_verbose(self):
        def simple_func():
            time.sleep(0.001)

        bench = Benchmark(warmup_runs=0, verbose=True)
        result = bench.run(simple_func, iterations=1)
        assert result is not None


class TestBenchmarkSuite:
    def test_add_and_get_result(self):
        suite = BenchmarkSuite("Test Suite")

        result1 = BenchmarkResult(
            name="test1", iterations=10, total_time=1.0,
            average_time=0.1, min_time=0.09, max_time=0.11,
            std_dev=0.01
        )
        suite.add("test1", result1)

        retrieved = suite.get_result("test1")
        assert retrieved is result1

    def test_compare(self):
        suite = BenchmarkSuite()

        result1 = BenchmarkResult(
            name="fast", iterations=10, total_time=1.0,
            average_time=0.1, min_time=0.09, max_time=0.11,
            std_dev=0.01
        )
        result2 = BenchmarkResult(
            name="slow", iterations=10, total_time=5.0,
            average_time=0.5, min_time=0.49, max_time=0.51,
            std_dev=0.01
        )

        suite.add("fast", result1)
        suite.add("slow", result2)

        comparison = suite.compare("average_time")
        assert comparison[0][0] == "fast"
        assert comparison[1][0] == "slow"

    def test_print_summary(self):
        suite = BenchmarkSuite("Test")

        result = BenchmarkResult(
            name="test", iterations=10, total_time=1.0,
            average_time=0.1, min_time=0.09, max_time=0.11,
            std_dev=0.01
        )
        suite.add("test", result)

        suite.print_summary()


class TestHelperFunctions:
    def test_benchmark_function(self):
        def func():
            x = torch.randn(10, 10)
            _ = x.sum()

        result = benchmark_function(func, iterations=2, warmup=0)
        assert isinstance(result, BenchmarkResult)

    def test_compare_functions(self):
        def func1():
            x = torch.randn(10, 10)
            _ = x.sum()

        def func2():
            x = torch.randn(10, 10)
            _ = x.mean()

        suite = compare_functions([("func1", func1), ("func2", func2)], iterations=2)
        assert len(suite.get_all_results()) == 2
