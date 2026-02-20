import pytest
import time
from attention_studio.utils.logging_utils import (
    PerformanceLogger,
    log_execution_time,
    log_function_call,
    DebugContext,
    get_performance_logger,
)


class TestPerformanceLogger:
    def test_start_and_end(self):
        logger = PerformanceLogger()
        logger.start("test_operation")
        time.sleep(0.01)
        logger.end("test_operation")

        timings = logger.get_timings("test_operation")
        assert len(timings) == 1
        assert timings[0].duration >= 0.01

    def test_measure_context_manager(self):
        logger = PerformanceLogger()
        with logger.measure("measured_op"):
            time.sleep(0.01)

        timings = logger.get_timings("measured_op")
        assert len(timings) == 1

    def test_get_total_time(self):
        logger = PerformanceLogger()
        logger.start("op1")
        time.sleep(0.01)
        logger.end("op1")
        logger.start("op1")
        time.sleep(0.01)
        logger.end("op1")

        total = logger.get_total_time("op1")
        assert total >= 0.02

    def test_get_average_time(self):
        logger = PerformanceLogger()
        for _ in range(3):
            logger.start("op")
            time.sleep(0.01)
            logger.end("op")

        avg = logger.get_average_time("op")
        assert avg >= 0.01

    def test_get_summary(self):
        logger = PerformanceLogger()
        logger.start("op1")
        logger.end("op1")
        logger.start("op2")
        logger.end("op2")

        summary = logger.get_summary()
        assert "op1" in summary
        assert "op2" in summary
        assert summary["op1"]["count"] == 1

    def test_clear(self):
        logger = PerformanceLogger()
        logger.start("op")
        logger.end("op")
        logger.clear()
        assert len(logger.get_timings()) == 0


class TestDecorators:
    def test_log_execution_time(self):
        @log_execution_time
        def slow_function():
            time.sleep(0.01)
            return 42

        result = slow_function()
        assert result == 42

    def test_log_function_call(self):
        @log_function_call
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3


class TestDebugContext:
    def test_context_enter_exit(self):
        with DebugContext("test_context"):
            pass

    def test_context_with_verbose_false(self):
        with DebugContext("test", verbose=False):
            pass


class TestGetPerformanceLogger:
    def test_singleton(self):
        logger1 = get_performance_logger()
        logger2 = get_performance_logger()
        assert logger1 is logger2
