"""The zoomable, pannable ROI canvas (a QGraphicsView).

Scene coordinates ARE source pixels: the frame is a pixmap item at 1:1, so
there is no display-to-source mapping error at any zoom. The canvas only
draws what the ``EditSession`` reports and forwards mouse / key input to it.
A polygon vertex (x, y) is the pixel (x, y); it is drawn at the pixel's
centre, hence the half-pixel between scene and model coordinates.
"""
import math
from typing import Callable, List, Optional, Tuple

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap,
                         QPolygonF, QTransform)
from PyQt6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem,
                             QGraphicsLineItem, QGraphicsPathItem,
                             QGraphicsPixmapItem, QGraphicsPolygonItem,
                             QGraphicsRectItem, QGraphicsScene,
                             QGraphicsSimpleTextItem, QGraphicsView)

from .rules import polygon_centroid
from .session import TOOL_DRAW, TOOL_EDIT, TOOL_LINE, EditSession

HALF_PIXEL = 0.5
MIN_ZOOM = 0.02
MAX_ZOOM = 64.0
Z_PIXMAP, Z_OVERLAY, Z_SHAPE, Z_LABEL, Z_HANDLE, Z_DRAFT = 0, 1, 2, 3, 4, 5

_CURSORS = {
    "arrow": Qt.CursorShape.ArrowCursor,
    "cross": Qt.CursorShape.CrossCursor,
    "vertex": Qt.CursorShape.PointingHandCursor,
    "move": Qt.CursorShape.SizeAllCursor,
    "rotate": Qt.CursorShape.PointingHandCursor,
}


def to_scene(point: Tuple[float, float]) -> QPointF:
    """Model (pixel-index) coordinates to scene coordinates."""
    return QPointF(point[0] + HALF_PIXEL, point[1] + HALF_PIXEL)


def to_model(scene_point: QPointF) -> Tuple[float, float]:
    return (scene_point.x() - HALF_PIXEL, scene_point.y() - HALF_PIXEL)


def _cosmetic_pen(color: QColor, width: float = 2.0,
                  style=Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(color, width, style)
    pen.setCosmetic(True)
    return pen


def _qcolor(rgb, alpha: int = 255) -> QColor:
    return QColor(rgb[0], rgb[1], rgb[2], alpha)


class RoiCanvas(QGraphicsView):
    """Draws the session's content over the video frame."""

    # Cursor position in source pixels (model coordinates) and whether it is
    # over the frame.
    pointer_moved = pyqtSignal(float, float, bool)
    pointer_left = pyqtSignal()
    zoom_changed = pyqtSignal(float)
    def __init__(self, session: EditSession, parent=None):
        super().__init__(parent)
        self.session = session
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QBrush(QColor(40, 40, 40)))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMinimumSize(320, 240)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._dynamic: List[QGraphicsItem] = []
        self._draft_items: List[QGraphicsItem] = []
        self._fit_mode = True
        self._space_down = False
        self._pan_origin: Optional[Tuple[int, int]] = None
        self._image_size = session.size
        # Called with a key event the canvas did not use; True if consumed.
        self.key_handler: Optional[Callable] = None

        session.changed.connect(self.refresh)
        session.state_changed.connect(self.refresh)
        self._apply_zoom_state()

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------
    def set_pixmap(self, pixmap: QPixmap, scale: Tuple[float, float] = (1.0, 1.0)) -> None:
        """Show ``pixmap`` so that scene units are source pixels.

        ``scale`` maps the pixmap's own pixels to source pixels (1, 1 for a
        full-resolution frame).
        """
        scene = self.scene()
        if self._pixmap_item is not None:
            scene.removeItem(self._pixmap_item)
        self._pixmap_item = scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(Z_PIXMAP)
        self._pixmap_item.setTransform(QTransform.fromScale(scale[0], scale[1]))
        width = pixmap.width() * scale[0]
        height = pixmap.height() * scale[1]
        self._image_size = (int(round(width)), int(round(height)))
        scene.setSceneRect(QRectF(0, 0, width, height))
        self.fit_to_view()
        self.refresh()

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    def image_rect(self) -> QRectF:
        return QRectF(0, 0, self._image_size[0], self._image_size[1])

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    @property
    def zoom(self) -> float:
        """Screen pixels per source pixel."""
        return self.transform().m11()

    def _apply_zoom_state(self) -> None:
        zoom = self.zoom
        self.session.view_scale = zoom
        if self._pixmap_item is not None:
            mode = (Qt.TransformationMode.SmoothTransformation if zoom < 1.0
                    else Qt.TransformationMode.FastTransformation)
            self._pixmap_item.setTransformationMode(mode)
        self.zoom_changed.emit(zoom)

    def set_zoom(self, factor: float) -> None:
        """Absolute zoom (1.0 = one screen pixel per source pixel)."""
        factor = min(max(float(factor), MIN_ZOOM), MAX_ZOOM)
        self._fit_mode = False
        self.setTransform(QTransform.fromScale(factor, factor))
        self._apply_zoom_state()
        self.refresh()

    def zoom_by(self, ratio: float) -> None:
        self.set_zoom(self.zoom * ratio)

    def zoom_actual(self) -> None:
        self.set_zoom(1.0)

    def fit_to_view(self) -> None:
        """Scale so the whole frame is visible; stays fitted on resize."""
        self._fit_mode = True
        rect = self.image_rect()
        # Before the widget is shown the viewport has no real size; showEvent
        # fits then, and until that zoom stays at 100%.
        if (rect.isEmpty() or not self.isVisible()
                or self.viewport().width() < 4 or self.viewport().height() < 4):
            return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        zoom = min(max(self.zoom, MIN_ZOOM), MAX_ZOOM)
        self.setTransform(QTransform.fromScale(zoom, zoom))
        self._apply_zoom_state()
        self.refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_to_view()

    def showEvent(self, event):
        super().showEvent(event)
        if self._fit_mode:
            self.fit_to_view()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                self.zoom_by(1.25 ** (delta / 120.0))
            event.accept()
            return
        super().wheelEvent(event)

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    def model_point(self, view_pos) -> Tuple[float, float]:
        """Model (source pixel) coordinates under a viewport position."""
        return to_model(self.mapToScene(view_pos))

    def view_pos_for_model(self, point: Tuple[float, float]) -> QPoint:
        """The viewport pixel showing a model coordinate (for tests)."""
        mapped = self.viewportTransform().map(to_scene(point))
        return QPoint(int(math.floor(mapped.x())), int(math.floor(mapped.y())))

    # ------------------------------------------------------------------
    # Mouse and keys
    # ------------------------------------------------------------------
    def mousePressEvent(self, event):
        self.setFocus()
        button = event.button()
        pos = event.position().toPoint()
        if button == Qt.MouseButton.MiddleButton or (
            button == Qt.MouseButton.LeftButton and self._space_down
        ):
            self._pan_origin = (pos.x(), pos.y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        self.session.press(self.model_point(pos), button, event.modifiers())
        event.accept()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._pan_origin is not None:
            dx, dy = pos.x() - self._pan_origin[0], pos.y() - self._pan_origin[1]
            self._pan_origin = (pos.x(), pos.y())
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - dx)
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - dy)
            event.accept()
            return
        point = self.model_point(pos)
        self.session.move(point, event.modifiers())
        inside = (0 <= point[0] <= self._image_size[0] - 1
                  and 0 <= point[1] <= self._image_size[1] - 1)
        self.pointer_moved.emit(point[0], point[1], inside)
        if not self._space_down:
            self.setCursor(_CURSORS[self.session.cursor_name(point)])
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._pan_origin is not None and event.button() in (
            Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton
        ):
            self._pan_origin = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._space_down
                           else Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.session.release(self.model_point(event.position().toPoint()),
                                 event.modifiers())
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.session.double_click(self.model_point(event.position().toPoint()),
                                      event.modifiers())
        event.accept()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.session.leave()
        self.pointer_left.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space:
            event.accept()
            return
        if self.key_handler is not None and self.key_handler(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            self._pan_origin = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self._space_down = False
        self._pan_origin = None
        super().focusOutEvent(event)

    # ------------------------------------------------------------------
    # Drawing the session
    # ------------------------------------------------------------------
    def _add(self, item: QGraphicsItem, z: float) -> QGraphicsItem:
        item.setZValue(z)
        self.scene().addItem(item)
        self._dynamic.append(item)
        return item

    def _handle(self, point, radius: float, fill: QColor, outline: QColor,
                square: bool = False, scene: bool = False) -> QGraphicsItem:
        rect = QRectF(-radius, -radius, 2 * radius, 2 * radius)
        item = QGraphicsRectItem(rect) if square else QGraphicsEllipseItem(rect)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        item.setBrush(QBrush(fill))
        item.setPen(QPen(outline, 1.5))
        item.setPos(QPointF(*point) if scene else to_scene(point))
        return self._add(item, Z_HANDLE)

    def _label(self, text: str, point, color: QColor, z: float = Z_LABEL,
               offset: Tuple[float, float] = (0.0, 0.0),
               scene: bool = False) -> QGraphicsItem:
        item = QGraphicsSimpleTextItem(text)
        item.setBrush(QBrush(QColor(255, 255, 255)))
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        bounds = item.boundingRect()
        backing = QGraphicsRectItem(bounds.adjusted(-3, -1, 3, 1), item)
        backing.setBrush(QBrush(QColor(color.red() // 2, color.green() // 2,
                                       color.blue() // 2, 190)))
        backing.setPen(QPen(Qt.PenStyle.NoPen))
        backing.setFlag(QGraphicsItem.GraphicsItemFlag.ItemStacksBehindParent)
        item.setPos((QPointF(*point) if scene else to_scene(point)) + QPointF(
            offset[0] - bounds.width() / 2.0, offset[1] - bounds.height() / 2.0
        ))
        return self._add(item, z)

    def refresh(self) -> None:
        """Rebuild every overlay item from the session's current state."""
        scene = self.scene()
        for item in self._dynamic:
            scene.removeItem(item)
        self._dynamic = []
        session = self.session
        document = session.document

        overlay = session.overlay
        if overlay is not None and len(overlay) >= 3:
            polygon = QPolygonF([to_scene(p) for p in overlay.tolist()])
            item = QGraphicsPolygonItem(polygon)
            item.setPen(_cosmetic_pen(QColor(0, 200, 200), 2, Qt.PenStyle.DashLine))
            item.setBrush(QBrush(QColor(0, 200, 200, 40)))
            self._add(item, Z_OVERLAY)

        counts = {}
        for shape in document.shapes:
            counts[shape.category] = counts.get(shape.category, 0) + 1
        seen = {}
        for shape in document.shapes:
            seen[shape.category] = seen.get(shape.category, 0) + 1
            rgb = session.category_color(shape.category)
            is_arena = shape.is_arena
            is_selected = shape.id == session.selected
            is_current = shape.category == session.current
            fill_alpha = 40 if is_arena else (110 if is_current else 70)
            item = QGraphicsPolygonItem(QPolygonF([to_scene(p) for p in shape.points]))
            if is_selected:
                pen = _cosmetic_pen(QColor(255, 255, 255), 3)
            elif is_arena:
                pen = _cosmetic_pen(_qcolor(rgb), 3)
            else:
                pen = _cosmetic_pen(_qcolor(rgb).darker(150), 3 if is_current else 2)
            item.setPen(pen)
            item.setBrush(QBrush(_qcolor(rgb, fill_alpha)))
            self._add(item, Z_SHAPE - (0.5 if is_arena else 0.0))

            name = session.description_for(shape.category)
            if is_arena:
                name = "Arena"
            if counts[shape.category] > 1:
                name = f"{name} #{seen[shape.category]}"
            centre = polygon_centroid(shape.points)
            self._label(name, centre, _qcolor(rgb))

            show_vertices = (is_selected and session.tool == TOOL_EDIT) or (
                session.tool == TOOL_DRAW
            )
            if show_vertices:
                big = is_selected and session.tool == TOOL_EDIT
                for index, vertex in enumerate(shape.points):
                    picked = is_selected and index == session.selected_vertex
                    self._handle(
                        vertex, 5.0 if big else 3.0,
                        QColor(255, 220, 0) if picked else QColor(255, 255, 255),
                        QColor(0, 0, 0),
                    )
            if is_selected and session.tool == TOOL_EDIT:
                self._add_transform_handles(shape)

        self._draw_line()
        self._draw_draft()

        if session.snap_target is not None:
            ring = QGraphicsEllipseItem(QRectF(-9, -9, 18, 18))
            ring.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            ring.setPen(QPen(QColor(255, 255, 0), 2))
            ring.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            ring.setPos(to_scene(session.snap_target))
            self._add(ring, Z_DRAFT)

    def _add_transform_handles(self, shape) -> None:
        handles = self.session.transform_handles(shape)
        xs = [p[0] for p in shape.points]
        top_centre = ((min(xs) + max(xs)) / 2.0, min(p[1] for p in shape.points))
        stem = QGraphicsLineItem()
        stem.setLine(to_scene(top_centre).x(), to_scene(top_centre).y(),
                     to_scene(handles["rotate"]).x(), to_scene(handles["rotate"]).y())
        stem.setPen(_cosmetic_pen(QColor(255, 255, 255, 160), 1, Qt.PenStyle.DashLine))
        self._add(stem, Z_HANDLE)
        self._handle(handles["rotate"], 5.0, QColor(120, 200, 255), QColor(0, 0, 0))
        self._handle(handles["scale"], 5.0, QColor(255, 160, 80), QColor(0, 0, 0),
                     square=True)

    def _draw_line(self) -> None:
        session = self.session
        line = session.document.line
        if line is None:
            return
        color = QColor(0, 230, 0)
        item = QGraphicsLineItem()
        item.setLine(line[0][0], line[0][1], line[1][0], line[1][1])
        item.setPen(_cosmetic_pen(color, 2))
        self._add(item, Z_SHAPE + 0.7)
        for end in line:
            self._handle(end, 4.0, color, QColor(0, 0, 0), scene=True)
        length = math.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1])
        middle = ((line[0][0] + line[1][0]) / 2.0, (line[0][1] + line[1][1]) / 2.0)
        self._label(f"{length:.2f} px", middle, color, offset=(0.0, -14.0), scene=True)

    def _draw_draft(self) -> None:
        session = self.session
        magenta = QColor(255, 0, 255)
        if session.tool == TOOL_LINE and session.line_start is not None:
            end = session.hover or session.line_start
            item = QGraphicsLineItem()
            b = to_scene(end)
            start = session.line_start
            item.setLine(start[0], start[1], b.x(), b.y())
            item.setPen(_cosmetic_pen(QColor(0, 230, 0), 2, Qt.PenStyle.DashLine))
            self._add(item, Z_DRAFT)
            self._handle(start, 4.0, QColor(0, 230, 0), QColor(0, 0, 0), scene=True)
            return
        if not session.draft:
            return
        path = QPainterPath()
        points = [to_scene(p) for p in session.draft]
        path.moveTo(points[0])
        for point in points[1:]:
            path.lineTo(point)
        item = QGraphicsPathItem(path)
        item.setPen(_cosmetic_pen(magenta, 2))
        self._add(item, Z_DRAFT)
        if len(points) > 2:
            closing = QGraphicsLineItem()
            closing.setLine(points[-1].x(), points[-1].y(), points[0].x(), points[0].y())
            closing.setPen(_cosmetic_pen(magenta, 1, Qt.PenStyle.DashLine))
            self._add(closing, Z_DRAFT)
        if session.hover is not None and session.tool == TOOL_DRAW:
            hover = to_scene(session.snap_target or session.hover)
            rubber = QGraphicsLineItem()
            rubber.setLine(points[-1].x(), points[-1].y(), hover.x(), hover.y())
            rubber.setPen(_cosmetic_pen(magenta, 1, Qt.PenStyle.DotLine))
            self._add(rubber, Z_DRAFT)
        for point in session.draft:
            self._handle(point, 3.5, QColor(255, 255, 255), magenta)


__all__ = ["RoiCanvas", "to_model", "to_scene"]
