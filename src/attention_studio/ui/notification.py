from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget


class NotificationType(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class NotificationPosition(Enum):
    TOP_RIGHT = "top_right"
    TOP_LEFT = "top_left"
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_LEFT = "bottom_left"
    TOP_CENTER = "top_center"
    BOTTOM_CENTER = "bottom_center"


@dataclass
class Notification:
    notification_id: str
    title: str
    message: str
    notification_type: NotificationType
    duration: int = 5000
    action_label: str | None = None
    action_callback: Callable | None = None


class ToastWidget(QWidget):
    closed = Signal(str)

    def __init__(self, notification: Notification, parent: QWidget | None = None):
        super().__init__(parent)
        self.notification = notification
        self.animation_progress = 0
        self.opacity = 0

        self.setFixedSize(320, 80)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._start_fade_in()

    def _start_fade_in(self):
        self.opacity = 0
        self.animation_progress = 0
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._fade_in_step)
        self._fade_timer.start(30)

    def _fade_in_step(self):
        self.opacity += 0.1
        if self.opacity >= 1:
            self.opacity = 1
            self._fade_timer.stop()
            if self.notification.duration > 0:
                QTimer.singleShot(self.notification.duration, self.close)
        self.update()

    def _fade_out_step(self):
        self.opacity -= 0.1
        if self.opacity <= 0:
            self.opacity = 0
            self._fade_timer.stop()
            self.close()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        color = self._get_color()
        painter.setBrush(QBrush(QColor(color)))
        painter.setPen(Qt.PenStyle.NoPen)

        painter.drawRoundedRect(self.rect(), 8, 8)

        title_color = "#FFFFFF"
        painter.setPen(QColor(title_color))
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        painter.setFont(font)
        painter.drawText(12, 22, self.notification.title)

        message_color = "#DDDDDD"
        painter.setPen(QColor(message_color))
        font_plain = QFont()
        font_plain.setPointSize(10)
        painter.setFont(font_plain)
        painter.drawText(12, 45, self.notification.message[:50])

        if self.notification.action_label:
            action_color = "#FFFFFF"
            painter.setPen(QColor(action_color))
            font_action = QFont()
            font_action.setPointSize(9)
            painter.setFont(font_action)
            painter.drawText(12, 65, f"  {self.notification.action_label}")

        painter.end()

    def _get_color(self) -> str:
        colors = {
            NotificationType.INFO: "#0078D4",
            NotificationType.SUCCESS: "#107C10",
            NotificationType.WARNING: "#FF8C00",
            NotificationType.ERROR: "#D13438",
        }
        base_color = colors.get(self.notification.notification_type, "#0078D4")
        return base_color

    def close(self):
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._fade_out_step)
        self._fade_timer.start(30)

    def mousePressEvent(self, event):
        if self.notification.action_callback:
            self.notification.action_callback()
        super().mousePressEvent(event)


class NotificationCenter:
    _instance: NotificationCenter | None = None

    def __new__(cls, parent: QWidget | None = None) -> NotificationCenter:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, parent: QWidget | None = None):
        if not self._initialized:
            self.parent = parent
            self.notifications: list[ToastWidget] = []
            self.notification_counter = 0
            self.position = NotificationPosition.TOP_RIGHT
            self.margin = 16
            self.spacing = 8
            self._initialized = True

    def set_position(self, position: NotificationPosition) -> None:
        self.position = position

    def show(
        self,
        title: str,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
        duration: int = 5000,
        action_label: str | None = None,
        action_callback: Callable | None = None,
    ) -> str:
        self.notification_counter += 1
        notification_id = f"notification_{self.notification_counter}"

        notification = Notification(
            notification_id=notification_id,
            title=title,
            message=message,
            notification_type=notification_type,
            duration=duration,
            action_label=action_label,
            action_callback=action_callback,
        )

        toast = ToastWidget(notification, self.parent)
        toast.closed.connect(self._on_toast_closed)

        self._position_toast(toast)
        toast.show()

        self.notifications.append(toast)

        return notification_id

    def info(self, title: str, message: str, **kwargs) -> str:
        return self.show(title, message, NotificationType.INFO, **kwargs)

    def success(self, title: str, message: str, **kwargs) -> str:
        return self.show(title, message, NotificationType.SUCCESS, **kwargs)

    def warning(self, title: str, message: str, **kwargs) -> str:
        return self.show(title, message, NotificationType.WARNING, **kwargs)

    def error(self, title: str, message: str, **kwargs) -> str:
        return self.show(title, message, NotificationType.ERROR, **kwargs)

    def _position_toast(self, toast: ToastWidget) -> None:
        if not self.parent:
            return

        parent_rect = self.parent.geometry()
        toast_width = toast.width()
        toast_height = toast.height()

        x, y = 0, 0

        if self.position in [NotificationPosition.TOP_RIGHT, NotificationPosition.TOP_LEFT]:
            y = self.margin
        elif self.position in [NotificationPosition.BOTTOM_RIGHT, NotificationPosition.BOTTOM_LEFT]:
            y = parent_rect.height() - toast_height - self.margin
        elif self.position == NotificationPosition.TOP_CENTER:
            y = self.margin
        elif self.position == NotificationPosition.BOTTOM_CENTER:
            y = parent_rect.height() - toast_height - self.margin

        if self.position in [NotificationPosition.TOP_RIGHT, NotificationPosition.BOTTOM_RIGHT]:
            x = parent_rect.width() - toast_width - self.margin
        elif self.position in [NotificationPosition.TOP_LEFT, NotificationPosition.BOTTOM_LEFT]:
            x = self.margin
        elif self.position in [NotificationPosition.TOP_CENTER, NotificationPosition.BOTTOM_CENTER]:
            x = (parent_rect.width() - toast_width) // 2

        y_offset = 0
        for existing_toast in self.notifications:
            if existing_toast.isVisible():
                y_offset += existing_toast.height() + self.spacing

        y += y_offset

        toast.move(x, y)

    def _on_toast_closed(self, notification_id: str):
        self.notifications = [t for t in self.notifications if t.notification.notification_id != notification_id]
        self._reposition_toasts()

    def _reposition_toasts(self):
        for toast in self.notifications:
            if toast.isVisible():
                self._position_toast(toast)

    def clear_all(self) -> None:
        for toast in self.notifications:
            toast.close()
        self.notifications = []

    def get_notification_count(self) -> int:
        return len([t for t in self.notifications if t.isVisible()])
