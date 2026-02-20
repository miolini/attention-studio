from __future__ import annotations

import functools
import logging
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


class LogCapture:
    def __init__(self):
        self.records: list = []

    def __enter__(self):
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.records.append(record)
        logging.root.addHandler(self.handler)
        return self

    def __exit__(self, *args):
        logging.root.removeHandler(self.handler)


@dataclass
class TimingRecord:
    name: str
    duration: float
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceLogger:
    def __init__(self, log_file: Path | None = None):
        self.log_file = log_file
        self._timings: list[TimingRecord] = []
        self._start_times: dict[str, float] = {}

    def start(self, name: str, metadata: dict | None = None):
        self._start_times[name] = time.perf_counter()

    def end(self, name: str, metadata: dict | None = None):
        if name not in self._start_times:
            logger.warning(f"Timer '{name}' was not started")
            return

        duration = time.perf_counter() - self._start_times[name]
        record = TimingRecord(
            name=name,
            duration=duration,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._timings.append(record)
        logger.debug(f"[TIMING] {name}: {duration:.4f}s")

    @contextmanager
    def measure(self, name: str, metadata: dict | None = None):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            record = TimingRecord(
                name=name,
                duration=duration,
                timestamp=time.time(),
                metadata=metadata or {},
            )
            self._timings.append(record)
            logger.debug(f"[TIMING] {name}: {duration:.4f}s")

    def get_timings(self, name: str | None = None) -> list[TimingRecord]:
        if name:
            return [t for t in self._timings if t.name == name]
        return self._timings

    def get_total_time(self, name: str) -> float:
        return sum(t.duration for t in self._timings if t.name == name)

    def get_average_time(self, name: str) -> float:
        timings = self.get_timings(name)
        if not timings:
            return 0.0
        return sum(t.duration for t in timings) / len(timings)

    def clear(self):
        self._timings.clear()
        self._start_times.clear()

    def get_summary(self) -> dict[str, dict[str, float]]:
        summary = {}
        for timing in self._timings:
            if timing.name not in summary:
                summary[timing.name] = {
                    "count": 0,
                    "total": 0,
                    "min": float("inf"),
                    "max": 0,
                }
            summary[timing.name]["count"] += 1
            summary[timing.name]["total"] += timing.duration
            summary[timing.name]["min"] = min(summary[timing.name]["min"], timing.duration)
            summary[timing.name]["max"] = max(summary[timing.name]["max"], timing.duration)

        for name in summary:
            count = summary[name]["count"]
            summary[name]["avg"] = summary[name]["total"] / count

        return summary


def log_execution_time(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.debug(f"Executed {func.__name__} in {duration:.4f}s")
        return result
    return wrapper


def log_function_call(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.trace(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.trace(f"Completed {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


class DebugContext:
    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            logger.debug(f"[ENTER] {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            logger.debug(f"[EXIT] {self.name}")
        return False


def setup_logger(
    log_file: Path | None = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
    )

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation=rotation,
            retention=retention,
        )


_performance_logger: PerformanceLogger | None = None


def get_performance_logger() -> PerformanceLogger:
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger


def configure_loguru(
    level: str = "INFO",
    format_string: str | None = None,
) -> None:
    if format_string is None:
        format_string = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <5}</level> | "
            "<level>{message}</level>"
        )

    logger.remove()
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
    )
