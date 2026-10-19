"""``RoiEditorWidget``: the embeddable ROI editor.

A toolbar (tools, undo/redo, zoom), the zone list, the zoomable canvas and a
status bar around one ``EditSession``. It reads no files and owns no dialog
buttons, so the v4.0 workspace can host it in a panel; ``RoiEditorDialog``
wraps it for the modal flow.
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QPushButton, QSplitter,
                             QStatusBar, QToolButton, QVBoxLayout, QWidget)

from roi_manager import ROIManager

from gui.qt_image import bgr_to_qpixmap

from .canvas import RoiCanvas
from .rules import STATUS_DONE, STATUS_MISSING, ZoneStatus
from .session import (TOOL_DRAW, TOOL_EDIT, TOOL_LINE, EditSession,
                      OverlayProvider)

LINE_KEY = "\0reference-line"
_DEFAULT_PROVIDER = object()

TOOL_LABELS = {
    TOOL_DRAW: ("Draw", "Draw a polygon: click to add points (D)"),
    TOOL_EDIT: ("Edit", "Move, reshape, rotate or scale polygons (E)"),
    TOOL_LINE: ("Line", "Draw or adjust the calibration reference line (L)"),
}


def default_overlay_provider(maze) -> Optional[OverlayProvider]:
    """The open-field center preview, for mazes that derive one.

    Only the area-share rule can be previewed while drawing: the distance rule
    needs the calibration, which comes from the reference line afterwards.
    """
    if maze is None or not hasattr(maze, "center_outline"):
        return None

    def provider(arena_points: np.ndarray) -> Optional[np.ndarray]:
        if not getattr(maze, "arena_enabled", False):
            return None
        if getattr(maze, "arena_center_mode", None) != "area_fraction":
            return None
        return maze.center_outline(arena_points, 0.0)

    return provider


class RoiEditorWidget(QWidget):
    """Edit ROI polygons, the arena and the reference line on a video frame.

    ``frame`` is the BGR frame at SOURCE resolution (or a QPixmap together
    with ``image_scale``, the pixmap-to-source factor). ``initial_roi_manager``
    preloads existing ROIs for editing.
    """

    # Something was drawn, moved, deleted, undone... (ROI content changed).
    changed = pyqtSignal()
    # (is_valid, list_of_issue_strings), only when the result changes.
    validity_changed = pyqtSignal(bool, list)

    def __init__(self, maze, frame: Union[np.ndarray, QPixmap],
                 parent: Optional[QWidget] = None,
                 initial_roi_manager: Optional[ROIManager] = None,
                 overlay_provider=_DEFAULT_PROVIDER,
                 image_scale: Tuple[float, float] = (1.0, 1.0),
                 line_only: bool = False):
        super().__init__(parent)
        self.maze = maze
        self._line_only = line_only
        pixmap, size = self._pixmap_and_size(frame, image_scale)
        provider = (default_overlay_provider(maze)
                    if overlay_provider is _DEFAULT_PROVIDER else overlay_provider)
        self.session = EditSession(maze, size, provider, self)
        self.canvas = RoiCanvas(self.session, self)
        self.canvas.key_handler = self.handle_key_event
        self._row_icons: Dict[Tuple[int, int, int], QIcon] = {}
        self._syncing_list = False
        self._build_ui()

        session = self.session
        session.changed.connect(self.changed)
        session.validity_changed.connect(self.validity_changed)
        session.changed.connect(self._refresh_ui)
        session.state_changed.connect(self._refresh_ui)
        session.undo_stack.canUndoChanged.connect(self._refresh_ui_flag)
        session.undo_stack.canRedoChanged.connect(self._refresh_ui_flag)
        self.canvas.pointer_moved.connect(self._on_pointer_moved)
        self.canvas.pointer_left.connect(self._on_pointer_left)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)

        self.canvas.set_pixmap(pixmap, image_scale)
        self.session.size = self.canvas.image_size
        if initial_roi_manager is not None:
            self.load_roi_manager(initial_roi_manager)
        else:
            self.session.load(None)
        self._refresh_ui()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @staticmethod
    def _pixmap_and_size(frame, image_scale):
        if isinstance(frame, QPixmap):
            pixmap = frame
        else:
            pixmap = bgr_to_qpixmap(frame)
        size = (int(round(pixmap.width() * image_scale[0])),
                int(round(pixmap.height() * image_scale[1])))
        return pixmap, size

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # -- toolbar ----------------------------------------------------
        toolbar = QHBoxLayout()
        self.tool_buttons: Dict[str, QToolButton] = {}
        for tool in (TOOL_DRAW, TOOL_EDIT, TOOL_LINE):
            if tool == TOOL_LINE and not self.session.line_enabled:
                continue
            if self._line_only and tool != TOOL_LINE:
                continue
            label, tip = TOOL_LABELS[tool]
            button = QToolButton()
            button.setText(label)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setAutoRaise(False)
            button.clicked.connect(lambda _checked=False, t=tool: self.set_tool(t))
            toolbar.addWidget(button)
            self.tool_buttons[tool] = button
        toolbar.addSpacing(12)
        self.undo_button = self._tool_button("Undo", "Undo (Ctrl+Z)", self.undo)
        self.redo_button = self._tool_button("Redo", "Redo (Ctrl+Shift+Z or Ctrl+Y)",
                                             self.redo)
        for button in (self.undo_button, self.redo_button):
            toolbar.addWidget(button)
        toolbar.addSpacing(12)
        self.fit_button = self._tool_button("Fit", "Fit the frame to the view (F)",
                                            self.zoom_fit)
        self.actual_button = self._tool_button("100%", "Actual size (1)",
                                               self.zoom_actual)
        for button in (self.fit_button, self.actual_button):
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        # -- zone panel + canvas ----------------------------------------
        self.instruction_label = QLabel()
        self.instruction_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.instruction_label.setWordWrap(True)
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            "color: #7a4b00; background: #fff3cd; padding: 4px 8px; "
            "border-radius: 3px;"
        )

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.instruction_label)
        right_layout.addWidget(self.hint_label)
        right_layout.addWidget(self.canvas, 1)

        self.zone_list = QListWidget()
        self.zone_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.zone_list.setIconSize(QSize(14, 14))
        self.zone_list.setWordWrap(True)
        self.zone_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.zone_list.setMinimumWidth(190)
        self.zone_list.currentRowChanged.connect(self._on_row_changed)
        self.zone_list.itemClicked.connect(self._on_item_clicked)
        self.delete_button = QPushButton("Delete selected")
        self.delete_button.setToolTip("Delete the selected vertex or polygon (Delete)")
        self.delete_button.setAutoDefault(False)
        self.delete_button.clicked.connect(self.delete_selection)
        self.clear_zone_button = QPushButton("Clear zone")
        self.clear_zone_button.setToolTip("Remove every polygon of the current zone")
        self.clear_zone_button.setAutoDefault(False)
        self.clear_zone_button.clicked.connect(self.clear_current_zone)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Zones"))
        left_layout.addWidget(self.zone_list, 1)
        left_layout.addWidget(self.delete_button)
        left_layout.addWidget(self.clear_zone_button)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left)
        left.setVisible(not self._line_only)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(1, 1)
        root.addWidget(self.splitter, 1)

        # -- status bar -------------------------------------------------
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.cursor_label = QLabel("x: -  y: -")
        self.zoom_label = QLabel("Zoom 100%")
        self.issue_label = QLabel()
        self.status_bar.addWidget(self.cursor_label)
        self.status_bar.addPermanentWidget(self.issue_label)
        self.status_bar.addPermanentWidget(self.zoom_label)
        root.addWidget(self.status_bar)

    @staticmethod
    def _tool_button(text: str, tip: str, slot: Callable) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.clicked.connect(lambda _checked=False: slot())
        return button

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def undo_stack(self):
        return self.session.undo_stack

    @property
    def source_size(self) -> Tuple[int, int]:
        return self.session.size

    def load_roi_manager(self, manager: Optional[ROIManager]) -> None:
        """Preload existing ROIs, the arena and the reference line for editing.

        Clears the undo history; the loaded state counts as saved.
        """
        self.session.load(manager)
        self._note_dimension_mismatch(manager)
        self._refresh_ui()

    def roi_manager(self) -> ROIManager:
        """A NEW ROIManager (int32 points, source dimensions set)."""
        return self.session.build_manager()

    def is_valid(self) -> bool:
        return self.session.is_valid()

    def issues(self) -> List[str]:
        return self.session.issues()

    def zone_status(self, category: str) -> ZoneStatus:
        return self.session.zone_status(category)

    def has_unsaved_work(self) -> bool:
        return self.session.has_unsaved_work()

    def mark_saved(self) -> None:
        self.session.mark_clean()

    def set_tool(self, tool: str) -> bool:
        return self.session.set_tool(tool)

    def tool(self) -> str:
        return self.session.tool

    def set_current_zone(self, category: str) -> bool:
        return self.session.activate_zone(category)

    def current_zone(self) -> Optional[str]:
        return self.session.current

    def undo(self) -> bool:
        return self.session.undo()

    def redo(self) -> bool:
        return self.session.redo()

    def delete_selection(self) -> bool:
        return self.session.delete_selection()

    def clear_current_zone(self) -> int:
        category = self.session.current
        return self.session.clear_zone(category) if category is not None else 0

    def add_point(self, x: float, y: float, snap: bool = True) -> bool:
        """Add a point (source pixels) to the polygon being drawn."""
        return self.session.add_draft_point((x, y), snap=snap)

    def close_shape(self) -> bool:
        """Close the polygon being drawn (same as Enter)."""
        return self.session.close_draft()

    def cancel_shape(self) -> bool:
        return self.session.cancel_draft()

    def pending_points(self) -> List[Tuple[int, int]]:
        return list(self.session.draft)

    def set_line(self, p1, p2) -> bool:
        """Set the reference line endpoints (source pixels) as one undo step."""
        return self.session.set_line((p1, p2), "Draw reference line")

    def line_points(self):
        return self.session.document.line

    def line_length(self) -> float:
        return self.session.line_length()

    def zoom_fit(self) -> None:
        self.canvas.fit_to_view()

    def zoom_actual(self) -> None:
        self.canvas.zoom_actual()

    def set_zoom(self, factor: float) -> None:
        self.canvas.set_zoom(factor)

    def zoom_percent(self) -> float:
        return self.canvas.zoom * 100.0

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------
    def handle_key_event(self, event) -> bool:
        """Editor shortcuts; True when the key was used."""
        key = event.key()
        modifiers = event.modifiers()
        if self.session.key_press(key, modifiers):
            return True
        ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        if ctrl:
            if key == Qt.Key.Key_0:
                self.zoom_fit()
                return True
            if key == Qt.Key.Key_1:
                self.zoom_actual()
                return True
            if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
                self.canvas.zoom_by(1.25)
                return True
            if key == Qt.Key.Key_Minus:
                self.canvas.zoom_by(0.8)
                return True
            return False
        tools = {Qt.Key.Key_D: TOOL_DRAW, Qt.Key.Key_E: TOOL_EDIT,
                 Qt.Key.Key_L: TOOL_LINE}
        if key in tools and not (modifiers & ~Qt.KeyboardModifier.ShiftModifier):
            self.set_tool(tools[key])
            return True
        if key == Qt.Key.Key_F:
            self.zoom_fit()
            return True
        if key == Qt.Key.Key_1:
            self.zoom_actual()
            return True
        return False

    # ------------------------------------------------------------------
    # Zone list
    # ------------------------------------------------------------------
    def _icon_for(self, rgb: Tuple[int, int, int]) -> QIcon:
        icon = self._row_icons.get(rgb)
        if icon is None:
            pixmap = QPixmap(14, 14)
            pixmap.fill(QColor(*rgb))
            icon = QIcon(pixmap)
            self._row_icons[rgb] = icon
        return icon

    @staticmethod
    def _status_text(row_description: str, status: ZoneStatus) -> Tuple[str, QColor]:
        count = status.count
        shapes = f"{count} polygon{'s' if count != 1 else ''}"
        if status.status == STATUS_DONE:
            return f"✓ {row_description}: {shapes}", QColor(30, 130, 50)
        if status.status == STATUS_MISSING:
            return f"○ {row_description}: missing", QColor(180, 110, 0)
        return f"✗ {row_description}: {status.issues[0].split(': ', 1)[-1]}", \
            QColor(190, 30, 30)

    def _refresh_zone_list(self) -> None:
        session = self.session
        rows = session.rows()
        wanted = [row.category for row in rows]
        has_line_row = session.line_enabled and not self._line_only
        if has_line_row:
            wanted.append(LINE_KEY)
        existing = [self.zone_list.item(i).data(Qt.ItemDataRole.UserRole)
                    for i in range(self.zone_list.count())]
        self._syncing_list = True
        try:
            if existing != wanted:
                self.zone_list.clear()
                for key in wanted:
                    item = QListWidgetItem()
                    item.setData(Qt.ItemDataRole.UserRole, key)
                    self.zone_list.addItem(item)
            for index, row in enumerate(rows):
                item = self.zone_list.item(index)
                status = session.zone_status(row.category)
                text, color = self._status_text(row.description, status)
                item.setText(text)
                item.setForeground(color)
                item.setIcon(self._icon_for(session.category_color(row.category)))
                item.setToolTip("\n".join(status.issues) or row.description)
                font = item.font()
                font.setBold(row.category == session.current
                             and session.tool != TOOL_LINE)
                item.setFont(font)
            if has_line_row:
                item = self.zone_list.item(len(rows))
                length = session.line_length()
                if length > 0:
                    text, color = f"✓ Reference line: {length:.1f} px", QColor(30, 130, 50)
                elif session.document.legacy_reference is not None:
                    text = (f"✓ Reference line: "
                            f"{session.document.legacy_reference.length:.1f} px (no ends)")
                    color = QColor(30, 130, 50)
                else:
                    text, color = "○ Reference line: optional", QColor(110, 110, 110)
                item.setText(text)
                item.setForeground(color)
                item.setIcon(self._icon_for((0, 230, 0)))
                font = item.font()
                font.setBold(session.tool == TOOL_LINE)
                item.setFont(font)
            # Highlight the current zone (or the line row).
            target = None
            if session.tool == TOOL_LINE and has_line_row:
                target = len(rows)
            elif session.current in [row.category for row in rows]:
                target = [row.category for row in rows].index(session.current)
            self.zone_list.setCurrentRow(target if target is not None else -1)
        finally:
            self._syncing_list = False

    def _on_row_changed(self, row: int) -> None:
        if self._syncing_list or row < 0:
            return
        self._activate_row(row)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self._activate_row(self.zone_list.row(item))

    def _activate_row(self, row: int) -> None:
        item = self.zone_list.item(row)
        if item is None:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        if key == LINE_KEY:
            self.session.set_tool(TOOL_LINE)
        else:
            self.session.activate_zone(key)
        self._refresh_ui()   # puts the highlight back if the switch was refused

    # ------------------------------------------------------------------
    # Keeping the widgets in step with the session
    # ------------------------------------------------------------------
    def _refresh_ui_flag(self, _flag: bool) -> None:
        self._refresh_ui()

    def _refresh_ui(self) -> None:
        session = self.session
        for tool, button in self.tool_buttons.items():
            button.setChecked(session.tool == tool)
        self.undo_button.setEnabled(
            session.undo_stack.canUndo() or bool(session.draft)
        )
        self.redo_button.setEnabled(
            session.undo_stack.canRedo() or bool(session.draft_redo)
        )
        self._refresh_zone_list()
        self.delete_button.setEnabled(
            session.selected is not None or (
                session.tool == TOOL_LINE and session.document.line is not None
            )
        )
        self.clear_zone_button.setEnabled(
            session.current is not None
            and bool(session.document.shapes_in(session.current))
        )
        self.instruction_label.setText(self._instruction())
        hint = session.hint()
        self.hint_label.setText(hint)
        self.hint_label.setVisible(bool(hint))
        issues = session.issues()
        if issues:
            self.issue_label.setText(f"{len(issues)} to fix")
            self.issue_label.setToolTip("\n".join(issues))
        else:
            self.issue_label.setText("All zones complete")
            self.issue_label.setToolTip("")

    def _instruction(self) -> str:
        session = self.session
        if session.tool == TOOL_LINE:
            return "Reference line: calibrate distances against a known length."
        if session.current is None:
            return ""
        description = session.description_for(session.current)
        verb = "Draw" if session.tool == TOOL_DRAW else "Edit"
        return f"{verb} polygon(s) for {description}."

    def _note_dimension_mismatch(self, manager: Optional[ROIManager]) -> None:
        dims = getattr(manager, "source_dimensions", None) if manager is not None else None
        if dims and tuple(dims) != tuple(self.session.size):
            self.session.show_message(
                f"These ROIs were defined for {dims[0]}x{dims[1]} px but this "
                f"frame is {self.session.size[0]}x{self.session.size[1]} px; "
                "check that they still line up."
            )

    def _on_pointer_moved(self, x: float, y: float, inside: bool) -> None:
        if inside:
            self.cursor_label.setText(f"x: {int(round(x))}  y: {int(round(y))} px")
        else:
            self.cursor_label.setText("x: -  y: -")

    def _on_pointer_left(self) -> None:
        self.cursor_label.setText("x: -  y: -")

    def _on_zoom_changed(self, zoom: float) -> None:
        self.zoom_label.setText(f"Zoom {zoom * 100:.0f}%")


__all__ = ["LINE_KEY", "RoiEditorWidget", "default_overlay_provider"]
