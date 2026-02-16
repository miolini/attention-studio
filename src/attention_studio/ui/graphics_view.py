from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView


class InteractiveGraphicsView(QGraphicsView):
    zoom_changed = Signal(float)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self._zoom = 1.0
        self._min_zoom = 0.01
        self._max_zoom = 50.0
        self._zoom_factor = 1.15
        self.setMouseTracking(True)

        self._panning = False
        self._pan_start_x = 0
        self._pan_start_y = 0

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)

        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setOptimizationFlags(
            QGraphicsView.DontSavePainterState |
            QGraphicsView.DontAdjustForAntialiasing
        )

        if scene:
            scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)

    def wheelEvent(self, event):
        zoom_factor = self._zoom_factor if event.angleDelta().y() > 0 else 1 / self._zoom_factor

        mouse_scene_pos = self.mapToScene(event.position().toPoint())

        new_zoom = self._zoom * zoom_factor
        if new_zoom < self._min_zoom or new_zoom > self._max_zoom:
            return

        self._zoom = new_zoom

        self.resetTransform()
        self.scale(self._zoom, self._zoom)

        mouse_viewport_pos = self.mapFromScene(mouse_scene_pos)
        h_scroll = self.horizontalScrollBar()
        v_scroll = self.verticalScrollBar()

        h_scroll.setValue(int(mouse_scene_pos.x() * self._zoom - mouse_viewport_pos.x()))
        v_scroll.setValue(int(mouse_scene_pos.y() * self._zoom - mouse_viewport_pos.y()))

        self.zoom_changed.emit(self._zoom)
        event.accept()

    def _zoom_in(self):
        new_zoom = self._zoom * self._zoom_factor
        if new_zoom <= self._max_zoom:
            center = self.viewport().rect().center()
            center_scene = self.mapToScene(center)
            self._zoom = new_zoom
            self.resetTransform()
            self.scale(self._zoom, self._zoom)
            new_center = self.mapFromScene(center_scene)
            self.horizontalScrollBar().setValue(int(new_center.x() - center.x()))
            self.verticalScrollBar().setValue(int(new_center.y() - center.y()))
            self.zoom_changed.emit(self._zoom)

    def _zoom_out(self):
        new_zoom = self._zoom / self._zoom_factor
        if new_zoom >= self._min_zoom:
            center = self.viewport().rect().center()
            center_scene = self.mapToScene(center)
            self._zoom = new_zoom
            self.resetTransform()
            self.scale(self._zoom, self._zoom)
            new_center = self.mapFromScene(center_scene)
            self.horizontalScrollBar().setValue(int(new_center.x() - center.x()))
            self.verticalScrollBar().setValue(int(new_center.y() - center.y()))
            self.zoom_changed.emit(self._zoom)

    def _apply_zoom(self):
        self.resetTransform()
        self.scale(self._zoom, self._zoom)
        self.zoom_changed.emit(self._zoom)

    def set_zoom(self, zoom_level):
        self._zoom = max(self._min_zoom, min(self._max_zoom, zoom_level))
        self._apply_zoom()

    def get_zoom(self):
        return self._zoom

    def fit_in_view(self):
        if self.scene():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
            self._zoom = self.transform().m11()
            self._zoom = max(self._min_zoom, min(self._max_zoom, self._zoom))
            self.zoom_changed.emit(self._zoom)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start_x = event.globalPosition().x()
            self._pan_start_y = event.globalPosition().y()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta_x = event.globalPosition().x() - self._pan_start_x
            delta_y = event.globalPosition().y() - self._pan_start_y
            self._pan_start_x = event.globalPosition().x()
            self._pan_start_y = event.globalPosition().y()

            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta_x)
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta_y)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self._zoom_in()
            event.accept()
        elif event.key() == Qt.Key_Minus:
            self._zoom_out()
            event.accept()
        elif event.key() == Qt.Key_0:
            self.fit_in_view()
            event.accept()
        else:
            super().keyPressEvent(event)
