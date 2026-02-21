from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StatusLevel(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class StatusMessage:
    message: str
    level: StatusLevel = StatusLevel.INFO
    timestamp: float = 0.0


@dataclass
class StatusBarState:
    is_loading: bool = False
    progress_value: int = 0
    progress_max: int = 100
    current_task: str = ""
    messages: list[StatusMessage] = field(default_factory=list)


class StatusBarController:
    def __init__(self):
        self.state = StatusBarState()
        self._callbacks: list[callable] = []

    def on_update(self, callback: callable) -> None:
        self._callbacks.append(callback)

    def _notify(self) -> None:
        for callback in self._callbacks:
            callback(self.state)

    def start_progress(
        self,
        task_name: str,
        max_value: int = 100,
    ) -> None:
        self.state.is_loading = True
        self.state.progress_max = max_value
        self.state.progress_value = 0
        self.state.current_task = task_name
        self._notify()

    def update_progress(self, value: int) -> None:
        self.state.progress_value = min(value, self.state.progress_max)
        self._notify()

    def increment_progress(self, amount: int = 1) -> None:
        self.state.progress_value = min(
            self.state.progress_value + amount,
            self.state.progress_max
        )
        self._notify()

    def finish_progress(self) -> None:
        self.state.is_loading = False
        self.state.progress_value = self.state.progress_max
        self.state.current_task = ""
        self._notify()

    def set_message(
        self,
        message: str,
        level: StatusLevel = StatusLevel.INFO,
    ) -> None:
        status_msg = StatusMessage(message=message, level=level)
        self.state.messages.append(status_msg)
        if len(self.state.messages) > 100:
            self.state.messages = self.state.messages[-100:]
        self._notify()

    def clear_messages(self) -> None:
        self.state.messages = []
        self._notify()

    def show_ready(self) -> None:
        self.set_message("Ready", StatusLevel.SUCCESS)

    def show_error(self, message: str) -> None:
        self.set_message(message, StatusLevel.ERROR)

    def show_warning(self, message: str) -> None:
        self.set_message(message, StatusLevel.WARNING)

    def show_info(self, message: str) -> None:
        self.set_message(message, StatusLevel.INFO)

    def get_current_status(self) -> str:
        if self.state.is_loading:
            percent = int(100 * self.state.progress_value / self.state.progress_max) if self.state.progress_max > 0 else 0
            return f"{self.state.current_task}... {percent}%"
        elif self.state.messages:
            return self.state.messages[-1].message
        else:
            return "Ready"

    def is_busy(self) -> bool:
        return self.state.is_loading

    def get_progress_percent(self) -> int:
        if not self.state.is_loading:
            return 0
        return int(100 * self.state.progress_value / self.state.progress_max) if self.state.progress_max > 0 else 0
