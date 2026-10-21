"""
gui/workspace/ui_kit.py
The reusable widgets of the 4.0 workspace design: small, themed, free of
business logic and testable on their own.

Colours always come from :func:`gui.workspace.theme.current_palette`, read at
paint time, so every widget follows a light/dark switch (``apply_theme``
repaints them through ``QEvent.PaletteChange``). Widgets that are plain Qt
controls (buttons, labels) are styled by the application style sheet through
the ``class`` / ``role`` dynamic properties.

Quick reference (details on each class)
---------------------------------------
Navigation      ``Stepper``, ``StageHeader``, ``FooterBar``
Buttons         ``PrimaryButton``, ``SecondaryButton``, ``GhostButton``,
                ``DangerButton``
Containers      ``Card``, ``CollapsibleSection``
Feedback        ``StatusChip``, ``Callout``, ``InlineHelp``, ``EmptyState``
Input           ``DropZone``, ``ParadigmCard`` / ``ParadigmPicker``,
                ``Filmstrip``
Small           ``KeyHint``, ``Label``
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from PyQt6.QtCore import (QAbstractAnimation, QAbstractListModel, QEvent,
                          QItemSelectionModel, QModelIndex, QPointF, QPropertyAnimation, QRect,
                          QRectF, QSize, Qt, QTimer, pyqtSignal)
from PyQt6.QtGui import (QColor, QFont, QFontMetrics, QIcon,
                         QPainter, QPainterPath, QPen, QPixmap)
from PyQt6.QtWidgets import (QAbstractButton, QAbstractItemView, QFrame,
                             QGridLayout, QHBoxLayout, QLabel, QListView,
                             QPushButton, QSizePolicy, QStyle,
                             QStyledItemDelegate, QToolTip, QVBoxLayout,
                             QWidget)

from gui.project_table_model import paths_from_mime
from gui.workspace.theme import (Palette, Radius, Spacing, TypeScale,
                                 current_palette, icon, icon_pixmap, mix, set_role)
from gui.workspace.thumbnails import ThumbnailLoader

__all__ = [
    "Callout", "Card", "CollapsibleSection", "DangerButton", "DropZone",
    "EmptyState", "Filmstrip", "FilmstripItem", "FooterBar", "GhostButton",
    "InlineHelp", "KeyHint", "Label", "PARADIGM_INFO", "ParadigmCard",
    "ParadigmPicker", "PrimaryButton", "SecondaryButton", "StageHeader",
    "StatusChip", "StepInfo", "Stepper", "STEP_STATUSES", "CHIP_KINDS",
]

CHIP_KINDS = ("ok", "warning", "error", "info", "neutral", "running")


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
class _Themed:
    """Mixin (first base): repaint when the theme changes."""

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange):
            self._theme_changed()

    def _theme_changed(self) -> None:
        self.update()


def _font(base: QFont, pixels: int, semibold: bool = False) -> QFont:
    font = QFont(base)
    font.setPixelSize(pixels)
    font.setWeight(QFont.Weight.DemiBold if semibold else QFont.Weight.Normal)
    return font


def _rounded(painter: QPainter, rect: QRectF, radius: float,
             fill: Optional[str], border: Optional[str] = None,
             border_width: float = 1.0) -> None:
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if border:
        half = border_width / 2.0
        rect = rect.adjusted(half, half, -half, -half)
        painter.setPen(QPen(QColor(border), border_width))
    else:
        painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(fill) if fill else Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(rect, radius, radius)
    painter.restore()


def _draw_icon(painter: QPainter, widget: QWidget, name: str, color: str,
               rect: QRectF) -> None:
    """Draw a line icon centred in ``rect`` (sharp at any screen density)."""
    side = max(1, round(min(rect.width(), rect.height())))
    pixmap = icon_pixmap(name, color, side, widget.devicePixelRatioF())
    painter.drawPixmap(
        QPointF(rect.center().x() - side / 2.0, rect.center().y() - side / 2.0),
        pixmap,
    )


def _elide(text: str, font: QFont, width: float) -> str:
    return QFontMetrics(font).elidedText(
        text, Qt.TextElideMode.ElideRight, max(0, int(width))
    )


def _focus_ring(painter: QPainter, rect: QRectF, radius: float, palette: Palette) -> None:
    _rounded(painter, rect.adjusted(-1, -1, 1, 1), radius + 1, None,
             palette.focus_ring, 2.0)


# ----------------------------------------------------------------------
# Labels and buttons
# ----------------------------------------------------------------------
class Label(QLabel):
    """A ``QLabel`` with a type-scale role: display, title, subtitle, body,
    caption or muted."""

    def __init__(self, text: str = "", role: str = "body", wrap: bool = False,
                 parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        set_role(self, role)
        self.setWordWrap(wrap)


class _ThemedButton(QPushButton):
    """Base of the four button helpers: sets the ``class`` property and keeps
    an optional icon in the colour that suits the variant."""

    VARIANT = "secondary"
    ICON_COLOR = "text"

    def __init__(self, text: str = "", icon_name: Optional[str] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self._icon_name: Optional[str] = None
        self.setProperty("class", self.VARIANT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if icon_name:
            self.set_icon_name(icon_name)

    def set_icon_name(self, name: Optional[str]) -> None:
        """Show the named line icon (or none) before the text."""
        self._icon_name = name
        self.setIcon(icon(name, self.ICON_COLOR) if name else QIcon())

    def icon_name(self) -> Optional[str]:
        return self._icon_name

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and self._icon_name:
            self.setIcon(icon(self._icon_name, self.ICON_COLOR))


class PrimaryButton(_ThemedButton):
    """The one main action of a screen (filled)."""
    VARIANT = "primary"
    ICON_COLOR = "primary_text"


class SecondaryButton(_ThemedButton):
    """A normal action (outlined)."""
    VARIANT = "secondary"


class GhostButton(_ThemedButton):
    """A quiet action without an outline (Back, Cancel, toolbar actions)."""
    VARIANT = "ghost"


class DangerButton(_ThemedButton):
    """A destructive action (outlined in red)."""
    VARIANT = "danger"
    ICON_COLOR = "danger"


# ----------------------------------------------------------------------
# Icon badge (icon on a soft coloured tile)
# ----------------------------------------------------------------------
class _IconBadge(_Themed, QWidget):
    """An icon on a softly tinted circle or rounded square."""

    def __init__(self, name: str, size: int = 40, color: str = "primary",
                 shape: str = "circle", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._name = name
        self._color = color
        self._shape = shape
        self.setFixedSize(size, size)

    def set_icon_name(self, name: str) -> None:
        self._name = name
        self.update()

    def icon_name(self) -> str:
        return self._name

    def paintEvent(self, event) -> None:
        palette = current_palette()
        rect = QRectF(self.rect())
        radius = rect.width() / 2 if self._shape == "circle" else rect.width() * 0.28
        painter = QPainter(self)
        _rounded(painter, rect, radius, palette.tint(self._color, 0.12))
        _draw_icon(painter, self, self._name, self._color,
                   QRectF(rect.center().x() - rect.width() * 0.25,
                          rect.center().y() - rect.height() * 0.25,
                          rect.width() * 0.5, rect.height() * 0.5))


# ----------------------------------------------------------------------
# Status chip
# ----------------------------------------------------------------------
_CHIP_ICONS = {"ok": "check", "warning": "warning", "error": "error", "info": "info"}


def _chip_size(text: str, font: QFont, has_glyph: bool) -> QSize:
    metrics = QFontMetrics(font)
    width = 16 + metrics.horizontalAdvance(text) + (16 if has_glyph else 0)
    return QSize(width, 22)


def _paint_chip(painter: QPainter, widget: QWidget, rect: QRectF, text: str,
                kind: str, font: QFont, spin_angle: float = 0.0) -> None:
    palette = current_palette()
    foreground, soft = palette.kind_colors(kind)
    _rounded(painter, rect, rect.height() / 2, soft)
    left = rect.left() + 8
    glyph = _CHIP_ICONS.get(kind)
    if glyph:
        _draw_icon(painter, widget, glyph, foreground,
                   QRectF(left, rect.center().y() - 6, 12, 12))
        left += 16
    elif kind == "running":
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(foreground), 1.8, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap))
        arc = QRectF(left + 1, rect.center().y() - 5, 10, 10)
        painter.drawArc(arc, int(-spin_angle * 16), 270 * 16)
        painter.restore()
        left += 16
    painter.setFont(font)
    painter.setPen(QColor(foreground))
    room = rect.right() - left - 8
    painter.drawText(
        QRectF(left, rect.top(), room, rect.height()),
        Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, _elide(text, font, room + 2),
    )


class StatusChip(_Themed, QWidget):
    """A small pill that states a status: ``kind`` is one of ``ok``,
    ``warning``, ``error``, ``info``, ``neutral`` or ``running`` (animated).

    The colour is never the only cue: ok / warning / error / info carry an
    icon and ``running`` a spinner.
    """

    def __init__(self, kind: str = "neutral", text: str = "",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._kind = "neutral"
        self._text = text
        self._angle = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._advance)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.set_kind(kind)

    def kind(self) -> str:
        return self._kind

    def text(self) -> str:
        return self._text

    def set_kind(self, kind: str) -> None:
        if kind not in CHIP_KINDS:
            raise ValueError(f"Unknown chip kind {kind!r}")
        self._kind = kind
        self._sync_timer()
        self.updateGeometry()
        self.update()

    def set_text(self, text: str) -> None:
        self._text = text
        self.setAccessibleName(text)
        self.updateGeometry()
        self.update()

    def set_status(self, kind: str, text: str) -> None:
        self._text = text
        self.set_kind(kind)
        self.setAccessibleName(text)

    def is_animating(self) -> bool:
        return self._timer.isActive()

    def _chip_font(self) -> QFont:
        return _font(self.font(), TypeScale.CAPTION, True)

    def sizeHint(self) -> QSize:
        glyph = self._kind in _CHIP_ICONS or self._kind == "running"
        return _chip_size(self._text, self._chip_font(), glyph)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def _sync_timer(self) -> None:
        if self._kind == "running" and self.isVisible():
            self._timer.start()
        else:
            self._timer.stop()

    def _advance(self) -> None:
        self._angle = (self._angle + 30) % 360
        self.update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._sync_timer()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        _paint_chip(painter, self, QRectF(self.rect()), self._text,
                    self._kind, self._chip_font(), self._angle)


# ----------------------------------------------------------------------
# Key hint
# ----------------------------------------------------------------------
class KeyHint(_Themed, QWidget):
    """A keyboard shortcut drawn as a key cap, e.g. ``KeyHint("S")``."""

    def __init__(self, text: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._text = text
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAccessibleName(f"Shortcut {text}")

    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text
        self.updateGeometry()
        self.update()

    def _key_font(self) -> QFont:
        return _font(self.font(), TypeScale.CAPTION, True)

    def sizeHint(self) -> QSize:
        width = QFontMetrics(self._key_font()).horizontalAdvance(self._text) + 14
        return QSize(max(22, width), 22)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        rect = QRectF(self.rect())
        cap = rect.adjusted(0, 0, 0, -2)
        _rounded(painter, rect.adjusted(0, 2, 0, 0), 5, palette.border)
        _rounded(painter, cap, 5, palette.surface, palette.border_strong, 1.0)
        painter.setFont(self._key_font())
        painter.setPen(QColor(palette.text))
        painter.drawText(cap, Qt.AlignmentFlag.AlignCenter, self._text)


# ----------------------------------------------------------------------
# Callout / inline help
# ----------------------------------------------------------------------
_CALLOUT_ICONS = {"info": "info", "tip": "info", "warning": "warning",
                  "error": "error", "success": "check-circle"}


class Callout(_Themed, QWidget):
    """A tinted box with an icon for a tip or a warning.

    ``kind``: ``info``, ``tip``, ``warning``, ``error`` or ``success``.
    ``text`` may contain simple rich text (``<b>``, ``<a>``).
    """

    def __init__(self, kind: str, text: str, title: str = "",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        if kind not in _CALLOUT_ICONS:
            raise ValueError(f"Unknown callout kind {kind!r}")
        self._kind = kind
        self._icon = QLabel(self)
        self._icon.setFixedSize(20, 20)
        self._title = Label(title, "subtitle", wrap=True)
        self._title.setVisible(bool(title))
        self._body = Label(text, "body", wrap=True)
        self._body.setTextFormat(Qt.TextFormat.AutoText)
        self._body.setOpenExternalLinks(True)
        self._body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(Spacing.XXS)
        column.addWidget(self._title)
        column.addWidget(self._body)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(Spacing.MD + 4, Spacing.SM, Spacing.MD, Spacing.SM)
        self._row.setSpacing(Spacing.SM)
        self._row.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignTop)
        self._row.addLayout(column, 1)
        self._row.setAlignment(column, Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self._refresh_icon()

    def kind(self) -> str:
        return self._kind

    def text(self) -> str:
        return self._body.text()

    def title(self) -> str:
        return self._title.text()

    def set_text(self, text: str) -> None:
        self._body.setText(text)

    def set_kind(self, kind: str) -> None:
        if kind not in _CALLOUT_ICONS:
            raise ValueError(f"Unknown callout kind {kind!r}")
        self._kind = kind
        self._refresh_icon()
        self.update()

    def _foreground(self) -> str:
        return current_palette().kind_colors(self._kind)[0]

    def _refresh_icon(self) -> None:
        self._icon.setPixmap(icon_pixmap(
            _CALLOUT_ICONS[self._kind], self._foreground(), 18,
            self.devicePixelRatioF(),
        ))

    def _theme_changed(self) -> None:
        self._refresh_icon()
        self.update()

    def paintEvent(self, event) -> None:
        palette = current_palette()
        foreground, soft = palette.kind_colors(self._kind)
        painter = QPainter(self)
        rect = QRectF(self.rect())
        _rounded(painter, rect, Radius.LG, soft, mix(foreground, palette.surface, 0.30))
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(foreground))
        clip = QPainterPath()
        clip.addRoundedRect(rect, Radius.LG, Radius.LG)
        painter.setClipPath(clip)
        painter.drawRect(QRectF(rect.left(), rect.top(), 4, rect.height()))
        painter.restore()



class InlineHelp(Callout):
    """A dismissible tip next to the control it explains (``kind`` info)."""

    dismissed = pyqtSignal()

    def __init__(self, text: str, dismissible: bool = False,
                 parent: Optional[QWidget] = None):
        super().__init__("tip", text, parent=parent)
        self._close = GhostButton("", "close")
        self._close.setFixedSize(24, 24)
        self._close.setToolTip("Dismiss")
        self._close.setAccessibleName("Dismiss")
        self._close.setVisible(dismissible)
        self._close.clicked.connect(self._dismiss)
        self._row.addWidget(self._close, 0, Qt.AlignmentFlag.AlignTop)

    @property
    def close_button(self) -> QPushButton:
        return self._close

    def _dismiss(self) -> None:
        self.hide()
        self.dismissed.emit()


# ----------------------------------------------------------------------
# Card and collapsible section
# ----------------------------------------------------------------------
class Card(_Themed, QWidget):
    """A surface with a border and rounded corners.

    Put content in ``card.body`` (a ``QVBoxLayout``) or use ``add_widget``.
    ``add_header_widget`` places a widget (e.g. a button) at the right of the
    title row.
    """

    def __init__(self, title: str = "", subtitle: str = "", padding: int = Spacing.MD,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(padding, padding, padding, padding)
        outer.setSpacing(Spacing.SM)
        self._header = QWidget(self)
        header_row = QHBoxLayout(self._header)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(Spacing.SM)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title_label = Label(title, "subtitle")
        self.subtitle_label = Label(subtitle, "muted", wrap=True)
        self.subtitle_label.setVisible(bool(subtitle))
        titles.addWidget(self.title_label)
        titles.addWidget(self.subtitle_label)
        header_row.addLayout(titles, 1)
        self._header_actions = QHBoxLayout()
        self._header_actions.setSpacing(Spacing.XS)
        header_row.addLayout(self._header_actions)
        self._header.setVisible(bool(title))
        outer.addWidget(self._header)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(Spacing.SM)
        outer.addLayout(self.body)

    def title(self) -> str:
        return self.title_label.text()

    def set_title(self, title: str, subtitle: Optional[str] = None) -> None:
        self.title_label.setText(title)
        if subtitle is not None:
            self.subtitle_label.setText(subtitle)
            self.subtitle_label.setVisible(bool(subtitle))
        self._header.setVisible(bool(title))

    @property
    def content_layout(self) -> QVBoxLayout:
        return self.body

    def add_widget(self, widget: QWidget, stretch: int = 0) -> None:
        self.body.addWidget(widget, stretch)

    def add_header_widget(self, widget: QWidget) -> None:
        self._header.setVisible(True)
        self._header_actions.addWidget(widget)

    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        _rounded(painter, QRectF(self.rect()), Radius.LG, palette.surface,
                 palette.border)


class CollapsibleSection(_Themed, QWidget):
    """A chevron header that shows or hides a body (progressive disclosure).

    ``body`` is a ``QVBoxLayout`` for the content. ``toggled(expanded)`` is
    emitted on every change. With ``settings`` (an ``AppSettings``) and
    ``settings_key`` the collapsed state is remembered between sessions;
    ``expanded`` is then only the default for a first run.
    """

    toggled = pyqtSignal(bool)

    def __init__(self, title: str, expanded: bool = False, summary: str = "",
                 settings=None, settings_key: str = "", animated: bool = True,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._settings = settings
        self._key = f"collapsed/{settings_key}" if settings is not None and settings_key else ""
        self._animated = animated
        if self._key:
            expanded = not settings.ui_flag(self._key, default=not expanded)
        self._expanded = expanded
        self._header = _ThemedButton(title)
        self._header.setProperty("class", "section")
        self._header.setCheckable(False)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.clicked.connect(self.toggle)
        self._summary = Label(summary, "muted")
        self._summary.setVisible(bool(summary))
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(Spacing.SM)
        head.addWidget(self._header, 1)
        head.addWidget(self._summary, 0, Qt.AlignmentFlag.AlignRight)
        self._body = QWidget(self)
        self.body = QVBoxLayout(self._body)
        self.body.setContentsMargins(Spacing.MD + Spacing.XXS, Spacing.XS, 0, Spacing.XS)
        self.body.setSpacing(Spacing.SM)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addLayout(head)
        outer.addWidget(self._body)
        self._animation = QPropertyAnimation(self._body, b"maximumHeight", self)
        self._animation.setDuration(140)
        self._animation.finished.connect(self._animation_done)
        self._body.setVisible(expanded)
        self._refresh_header()

    # -- state ------------------------------------------------------------
    def title(self) -> str:
        return self._header.text()

    def is_expanded(self) -> bool:
        return self._expanded

    def set_summary(self, text: str) -> None:
        """A short muted note at the right of the header (e.g. "3 changed")."""
        self._summary.setText(text)
        self._summary.setVisible(bool(text))

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        if expanded == self._expanded:
            return
        self._expanded = expanded
        if self._key:
            self._settings.set_ui_flag(self._key, not expanded)
        self._refresh_header()
        self._animation.stop()
        if self._animated and self.isVisible():
            self._start_animation(expanded)
        else:
            self._body.setMaximumHeight(16777215)
            self._body.setVisible(expanded)
        self.toggled.emit(expanded)

    def animation_running(self) -> bool:
        return self._animation.state() == QAbstractAnimation.State.Running

    def _start_animation(self, expanded: bool) -> None:
        full = max(1, self._body.sizeHint().height())
        self._body.setVisible(True)
        self._animation.setStartValue(0 if expanded else self._body.height())
        self._animation.setEndValue(full if expanded else 0)
        self._body.setMaximumHeight(0 if expanded else self._body.height())
        self._animation.start()

    def _animation_done(self) -> None:
        self._body.setMaximumHeight(16777215)
        self._body.setVisible(self._expanded)

    def _refresh_header(self) -> None:
        self._header.set_icon_name("chevron-down" if self._expanded else "chevron-right")
        self._header.setAccessibleDescription("expanded" if self._expanded else "collapsed")


# ----------------------------------------------------------------------
# Empty state
# ----------------------------------------------------------------------
class EmptyState(_Themed, QWidget):
    """A friendly "nothing here yet" panel: big icon, title, one or two
    sentences and up to two actions (``add_action``)."""

    MAX_ACTIONS = 2

    def __init__(self, icon_name: str, title: str, body: str = "",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._badge = _IconBadge(icon_name, 72)
        self.title_label = Label(title, "title")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.title_label.setWordWrap(True)
        self.body_label = Label(body, "muted", wrap=True)
        self.body_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.body_label.setMaximumWidth(460)
        self.body_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.body_label.setVisible(bool(body))
        self._body_row = QHBoxLayout()
        self._body_row.setContentsMargins(0, 0, 0, 0)
        self._body_row.addStretch(1)
        self._body_row.addWidget(self.body_label, 4)
        self._body_row.addStretch(1)
        self._actions = QHBoxLayout()
        self._actions.setSpacing(Spacing.SM)
        self.buttons: List[QPushButton] = []
        column = QVBoxLayout(self)
        column.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        column.setSpacing(Spacing.SM)
        column.addStretch(1)
        column.addWidget(self._badge, 0, Qt.AlignmentFlag.AlignHCenter)
        column.addSpacing(Spacing.XS)
        column.addWidget(self.title_label)
        column.addLayout(self._body_row)
        column.addSpacing(Spacing.SM)
        column.addLayout(self._actions)
        column.addStretch(2)
        self._actions.setAlignment(Qt.AlignmentFlag.AlignHCenter)

    def title(self) -> str:
        return self.title_label.text()

    def body(self) -> str:
        return self.body_label.text()

    def set_texts(self, title: str, body: str = "") -> None:
        self.title_label.setText(title)
        self.body_label.setText(body)
        self.body_label.setVisible(bool(body))

    def set_icon_name(self, name: str) -> None:
        self._badge.set_icon_name(name)

    def add_action(self, text: str, primary: Optional[bool] = None,
                   icon_name: Optional[str] = None) -> QPushButton:
        """Add an action button; the first is primary unless told otherwise."""
        if len(self.buttons) >= self.MAX_ACTIONS:
            raise ValueError("An empty state offers at most two actions")
        if primary is None:
            primary = not self.buttons
        button = (PrimaryButton if primary else SecondaryButton)(text, icon_name)
        self.buttons.append(button)
        self._actions.addWidget(button)
        return button

    def clear_actions(self) -> None:
        while self.buttons:
            button = self.buttons.pop()
            self._actions.removeWidget(button)
            button.deleteLater()


# ----------------------------------------------------------------------
# Stage header and footer
# ----------------------------------------------------------------------
class StageHeader(_Themed, QWidget):
    """The top of a stage: icon, title, a one-sentence "what to do here", an
    optional help button (``help_requested``) and a slot for actions."""

    help_requested = pyqtSignal()

    def __init__(self, title: str, subtitle: str = "", icon_name: str = "info",
                 show_help: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._badge = _IconBadge(icon_name, 44, shape="rounded")
        self.title_label = Label(title, "title")
        self.subtitle_label = Label(subtitle, "muted", wrap=True)
        self.subtitle_label.setVisible(bool(subtitle))
        texts = QVBoxLayout()
        texts.setSpacing(2)
        texts.addWidget(self.title_label)
        texts.addWidget(self.subtitle_label)
        self._actions = QHBoxLayout()
        self._actions.setSpacing(Spacing.XS)
        self.help_button = GhostButton("", "help")
        self.help_button.setFixedSize(34, 34)
        self.help_button.setToolTip("What is this step?")
        self.help_button.setAccessibleName("Help")
        self.help_button.clicked.connect(self.help_requested)
        self.help_button.setVisible(show_help)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(Spacing.MD)
        row.addWidget(self._badge, 0, Qt.AlignmentFlag.AlignTop)
        row.addLayout(texts, 1)
        row.addLayout(self._actions)
        row.addWidget(self.help_button, 0, Qt.AlignmentFlag.AlignTop)

    def title(self) -> str:
        return self.title_label.text()

    def subtitle(self) -> str:
        return self.subtitle_label.text()

    def set_texts(self, title: str, subtitle: str = "") -> None:
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)
        self.subtitle_label.setVisible(bool(subtitle))

    def set_icon_name(self, name: str) -> None:
        self._badge.set_icon_name(name)

    def set_help_visible(self, visible: bool) -> None:
        self.help_button.setVisible(visible)

    @property
    def actions_layout(self) -> QHBoxLayout:
        """The right-aligned slot for action buttons."""
        return self._actions

    def add_action(self, widget: QWidget) -> None:
        self._actions.addWidget(widget, 0, Qt.AlignmentFlag.AlignTop)


class FooterBar(_Themed, QWidget):
    """The bottom bar of a stage: "← Back", a status in the middle and the
    primary "Next: <label> →" button.

    ``set_next("Detection", enabled=False, reason="Add a video first")`` keeps
    the button visible but disabled, with the reason as its tooltip.
    """

    back_clicked = pyqtSignal()
    next_clicked = pyqtSignal()

    BACK_TEXT = "← Back"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.back_button = GhostButton(self.BACK_TEXT)
        self.back_button.clicked.connect(self.back_clicked)
        self.chip = StatusChip("neutral", "")
        self.chip.setVisible(False)
        self.status_label = Label("", "muted")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._next_text = "Next →"
        self.next_button = PrimaryButton(self._next_text)
        self.next_button.setMinimumWidth(150)
        self.next_button.clicked.connect(self.next_clicked)
        self._extras = QHBoxLayout()
        self._extras.setSpacing(Spacing.XS)
        row = QHBoxLayout(self)
        row.setContentsMargins(Spacing.LG, Spacing.SM, Spacing.LG, Spacing.SM)
        row.setSpacing(Spacing.SM)
        row.addWidget(self.back_button)
        row.addSpacing(Spacing.SM)
        row.addWidget(self.chip)
        row.addWidget(self.status_label, 1)
        row.addLayout(self._extras)
        row.addWidget(self.next_button)

    def set_next(self, label: Optional[str], enabled: bool = True, reason: str = "",
                 prefix: str = "Next: ", arrow: bool = True) -> None:
        """Set the primary button. ``label=None`` hides it (last stage)."""
        if label is None:
            self.next_button.setVisible(False)
            return
        self.next_button.setVisible(True)
        self._next_text = f"{prefix}{label}{' →' if arrow else ''}"
        # "&" would make the next letter a mnemonic ("Arena & zones").
        self.next_button.setText(self._next_text.replace("&", "&&"))
        self.next_button.setEnabled(enabled)
        self.next_button.setToolTip("" if enabled else reason)

    def next_text(self) -> str:
        """The primary button's text as set (``next_button.text()`` doubles
        any "&" so Qt does not read it as a shortcut marker)."""
        return self._next_text

    def set_back(self, visible: bool = True, enabled: bool = True) -> None:
        self.back_button.setVisible(visible)
        self.back_button.setEnabled(enabled)

    def set_status(self, text: str, chip: Optional[Tuple[str, str]] = None) -> None:
        """The muted text in the middle, with an optional ``(kind, text)`` chip."""
        self.status_label.setText(text)
        self.status_label.setToolTip(text)
        if chip is None:
            self.chip.setVisible(False)
        else:
            self.chip.set_status(*chip)
            self.chip.setVisible(True)

    def add_extra(self, widget: QWidget) -> None:
        """A secondary button left of the primary one (e.g. "Save")."""
        self._extras.addWidget(widget)

    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(palette.surface))
        painter.fillRect(QRect(0, 0, self.width(), 1), QColor(palette.border))


# ----------------------------------------------------------------------
# Stepper
# ----------------------------------------------------------------------
STEP_STATUSES = ("todo", "current", "done", "attention", "disabled")
_STATUS_TEXT = {
    "todo": "Not started", "current": "You are here", "done": "Done",
    "attention": "Needs attention", "disabled": "Not available yet",
}


@dataclass
class StepInfo:
    """One step of a ``Stepper``."""
    id: str
    label: str
    icon: str = "info"
    status: str = "todo"
    reason: str = ""


@dataclass
class _Cell:
    rect: QRectF
    badge: QRectF
    text: QRectF
    mode: str                    # "full", "label" or "badge"


class Stepper(_Themed, QWidget):
    """The top navigation: numbered steps joined by connector lines.

    Each step has an icon, a label and a status (``todo``, ``current``,
    ``done``, ``attention`` or ``disabled``) plus a ``reason`` shown in its
    tooltip. Steps are clickable (not the disabled ones): ``step_clicked(id)``.
    Six steps fit 1100 px with captions; narrower, the captions go first and
    then the labels are elided; below ~100 px per step only the badges stay.
    """

    step_clicked = pyqtSignal(str)

    def __init__(self, steps: Sequence[Union[StepInfo, tuple]] = (),
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._steps: List[StepInfo] = []
        self._hover = -1
        self._pressed = -1
        self._focus = -1
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("Workflow steps")
        self.set_steps(steps)

    # -- content ----------------------------------------------------------
    def set_steps(self, steps: Sequence[Union[StepInfo, tuple]]) -> None:
        self._steps = [s if isinstance(s, StepInfo) else StepInfo(*s) for s in steps]
        for step in self._steps:
            self._check_status(step.status)
        self._focus = -1
        self.updateGeometry()
        self.update()

    def add_step(self, step_id: str, label: str, icon_name: str = "info",
                 status: str = "todo", reason: str = "") -> None:
        self._check_status(status)
        self._steps.append(StepInfo(step_id, label, icon_name, status, reason))
        self.updateGeometry()
        self.update()

    @staticmethod
    def _check_status(status: str) -> None:
        if status not in STEP_STATUSES:
            raise ValueError(f"Unknown step status {status!r}")

    def step_ids(self) -> List[str]:
        return [step.id for step in self._steps]

    def _find(self, step_id: str) -> int:
        for index, step in enumerate(self._steps):
            if step.id == step_id:
                return index
        raise KeyError(step_id)

    def status(self, step_id: str) -> str:
        return self._steps[self._find(step_id)].status

    def reason(self, step_id: str) -> str:
        return self._steps[self._find(step_id)].reason

    def label(self, step_id: str) -> str:
        return self._steps[self._find(step_id)].label

    def set_status(self, step_id: str, status: str, reason: Optional[str] = None) -> None:
        """Change a step's status (and, if given, its tooltip reason)."""
        self._check_status(status)
        step = self._steps[self._find(step_id)]
        step.status = status
        if reason is not None:
            step.reason = reason
        self.update()

    def current_id(self) -> Optional[str]:
        for step in self._steps:
            if step.status == "current":
                return step.id
        return None

    def set_current(self, step_id: str) -> None:
        """Make a step the current one (the previous current becomes todo)."""
        index = self._find(step_id)
        for other in self._steps:
            if other.status == "current":
                other.status = "todo"
        self._steps[index].status = "current"
        self.update()

    def tooltip_for(self, step_id: str) -> str:
        step = self._steps[self._find(step_id)]
        text = f"{step.label}: {_STATUS_TEXT[step.status]}"
        return f"{text}\n{step.reason}" if step.reason else text

    # -- geometry ---------------------------------------------------------
    def sizeHint(self) -> QSize:
        return QSize(max(480, 150 * len(self._steps)), 60)

    def minimumSizeHint(self) -> QSize:
        return QSize(44 * max(1, len(self._steps)), 60)

    def _layout(self) -> List[_Cell]:
        count = len(self._steps)
        if count == 0:
            return []
        width = float(self.width())
        margin = 4.0
        label_metrics = QFontMetrics(self._label_font())
        widest = max(
            [label_metrics.horizontalAdvance(step.label) for step in self._steps]
            + [QFontMetrics(self._caption_font()).horizontalAdvance("STEP 0")]
        )
        natural = 10.0 + 28.0 + 10.0 + widest + 16.0
        min_gap = 14.0
        spare = width - 2 * margin - count * natural
        if count == 1 or spare >= (count - 1) * min_gap:
            # Roomy: natural-size steps, connector lines take the slack.
            cell_width = natural
            gap = min(96.0, spare / (count - 1)) if count > 1 else 0.0
            margin = (width - count * cell_width - (count - 1) * gap) / 2.0
        else:
            gap = 12.0 if width >= 600 else 6.0
            cell_width = (width - 2 * margin - (count - 1) * gap) / count
        height = min(float(self.height()) - 8.0, 52.0)
        top = (self.height() - height) / 2.0
        mode = "full" if cell_width >= 138 else "label" if cell_width >= 100 else "badge"
        badge = 28.0
        cells: List[_Cell] = []
        for index in range(count):
            left = margin + index * (cell_width + gap)
            rect = QRectF(left, top, cell_width, height)
            if mode == "badge":
                badge_rect = QRectF(rect.center().x() - badge / 2, rect.center().y() - badge / 2,
                                    badge, badge)
                text = QRectF()
            else:
                badge_rect = QRectF(rect.left() + 10, rect.center().y() - badge / 2, badge, badge)
                text = QRectF(badge_rect.right() + 10, rect.top(),
                              rect.right() - badge_rect.right() - 18, rect.height())
            cells.append(_Cell(rect, badge_rect, text, mode))
        return cells

    def layout_mode(self) -> str:
        """``"full"`` (captions and labels), ``"label"`` or ``"badge"``."""
        cells = self._layout()
        return cells[0].mode if cells else "full"

    def step_rect(self, step_id: str) -> QRectF:
        return self._layout()[self._find(step_id)].rect

    def badge_rect(self, step_id: str) -> QRectF:
        """The round status badge of a step (widget coordinates)."""
        return self._layout()[self._find(step_id)].badge

    def displayed_label(self, step_id: str) -> str:
        """The label as drawn (elided when the step is too narrow)."""
        index = self._find(step_id)
        cell = self._layout()[index]
        if cell.mode == "badge":
            return ""
        return _elide(self._steps[index].label, self._label_font(), cell.text.width())

    def _index_at(self, point) -> int:
        for index, cell in enumerate(self._layout()):
            if cell.rect.adjusted(-3, -4, 3, 4).contains(QPointF(point)):
                return index
        return -1

    def _label_font(self) -> QFont:
        return _font(self.font(), TypeScale.BODY, True)

    def _caption_font(self) -> QFont:
        return _font(self.font(), TypeScale.CAPTION, False)

    # -- interaction ------------------------------------------------------
    def _enabled_index(self, index: int) -> bool:
        return 0 <= index < len(self._steps) and self._steps[index].status != "disabled"

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            index = self._index_at(event.pos())
            if index >= 0:
                step = self._steps[index]
                QToolTip.showText(event.globalPos(), self.tooltip_for(step.id), self,
                                  self._layout()[index].rect.toRect())
            else:
                QToolTip.hideText()
            return True
        return super().event(event)

    def mouseMoveEvent(self, event) -> None:
        index = self._index_at(event.position())
        if index != self._hover:
            self._hover = index
            self.setCursor(Qt.CursorShape.PointingHandCursor if self._enabled_index(index)
                           else Qt.CursorShape.ArrowCursor)
            self.update()

    def leaveEvent(self, event) -> None:
        self._hover = -1
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = self._index_at(event.position())

    def mouseReleaseEvent(self, event) -> None:
        index = self._index_at(event.position())
        pressed, self._pressed = self._pressed, -1
        if event.button() == Qt.MouseButton.LeftButton and index == pressed \
                and self._enabled_index(index):
            self._focus = index
            self.step_clicked.emit(self._steps[index].id)
            self.update()

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        if not self._enabled_index(self._focus):
            current = [i for i, s in enumerate(self._steps) if s.status == "current"]
            self._focus = current[0] if current else next(
                (i for i in range(len(self._steps)) if self._enabled_index(i)), -1)
        self.update()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        step = {Qt.Key.Key_Right: 1, Qt.Key.Key_Down: 1,
                Qt.Key.Key_Left: -1, Qt.Key.Key_Up: -1}.get(key)
        if step:
            index = self._focus + step
            while 0 <= index < len(self._steps) and not self._enabled_index(index):
                index += step
            if self._enabled_index(index):
                self._focus = index
                self.update()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space) \
                and self._enabled_index(self._focus):
            self.step_clicked.emit(self._steps[self._focus].id)
            return
        super().keyPressEvent(event)

    def focused_id(self) -> Optional[str]:
        return self._steps[self._focus].id if self._enabled_index(self._focus) else None

    # -- painting ---------------------------------------------------------
    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cells = self._layout()
        for index in range(len(cells) - 1):
            done = self._steps[index].status == "done"
            line = QRectF(cells[index].rect.right() + 3, cells[index].rect.center().y() - 1,
                          cells[index + 1].rect.left() - cells[index].rect.right() - 6, 2)
            _rounded(painter, line, 1, palette.success if done else palette.border)
        for index, (step, cell) in enumerate(zip(self._steps, cells)):
            self._paint_step(painter, palette, index, step, cell)

    def _paint_step(self, painter: QPainter, palette: Palette, index: int,
                    step: StepInfo, cell: _Cell) -> None:
        status = step.status
        current = status == "current"
        disabled = status == "disabled"
        if current:
            _rounded(painter, cell.rect, 12, palette.tint("primary", 0.12),
                     palette.primary, 1.5)
        elif index == self._hover and not disabled:
            _rounded(painter, cell.rect, 12, palette.surface_alt)
        if self.hasFocus() and index == self._focus:
            _focus_ring(painter, cell.rect, 12, palette)
        self._paint_badge(painter, palette, step, cell.badge, index)
        if cell.mode == "badge":
            return
        label_color = (palette.text_disabled if disabled
                       else palette.primary if current else palette.text)
        label_font = self._label_font()
        text = _elide(step.label, label_font, cell.text.width())
        if cell.mode == "full":
            caption_font = self._caption_font()
            caption_color = palette.text_disabled if disabled else (
                palette.primary if current else palette.text_muted)
            caption_height = QFontMetrics(caption_font).height()
            label_height = QFontMetrics(label_font).height()
            top = cell.rect.center().y() - (caption_height + label_height) / 2.0
            painter.setFont(caption_font)
            painter.setPen(QColor(caption_color))
            painter.drawText(QRectF(cell.text.left(), top, cell.text.width(), caption_height),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             f"STEP {index + 1}")
            painter.setFont(label_font)
            painter.setPen(QColor(label_color))
            painter.drawText(QRectF(cell.text.left(), top + caption_height,
                                    cell.text.width(), label_height),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
        else:
            painter.setFont(label_font)
            painter.setPen(QColor(label_color))
            painter.drawText(cell.text, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             text)

    def _paint_badge(self, painter: QPainter, palette: Palette, step: StepInfo,
                     rect: QRectF, index: int) -> None:
        status = step.status
        if status == "current":
            fill, border, color, glyph = palette.primary, palette.primary, palette.primary_text, step.icon
        elif status == "done":
            fill, border, color, glyph = palette.tint("success", 0.16), palette.success, palette.success, "check"
        elif status == "attention":
            fill, border, color, glyph = palette.tint("warning", 0.16), palette.warning, palette.warning, "warning"
        elif status == "disabled":
            fill, border, color, glyph = palette.surface, palette.border, palette.text_disabled, step.icon
        else:
            fill, border, color, glyph = palette.surface, palette.border_strong, palette.text_muted, step.icon
        _rounded(painter, rect, rect.width() / 2, fill, border, 1.5)
        _draw_icon(painter, self, glyph, color, rect.adjusted(6, 6, -6, -6))


# ----------------------------------------------------------------------
# Drop zone
# ----------------------------------------------------------------------
class DropZone(_Themed, QWidget):
    """A large dashed area that accepts dropped videos and folders.

    Signals: ``paths_dropped(list[Path])`` (videos, folders already expanded),
    ``paths_ignored(list[Path])`` (dropped but not videos),
    ``browse_requested()`` / ``folder_requested()`` (clicks on the zone or its
    buttons: the owner opens the file or folder dialog).
    ``set_recursive(True)`` searches dropped folders recursively.
    """

    paths_dropped = pyqtSignal(list)
    paths_ignored = pyqtSignal(list)
    browse_requested = pyqtSignal()
    folder_requested = pyqtSignal()

    def __init__(self, title: str = "Drop videos or folders here",
                 hint: str = "MP4, AVI, MOV or MKV. Folders are searched for videos.",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dragging = False
        self._hover = False
        self._recursive = False
        self._compact = False
        self._badge = _IconBadge("upload", 64)
        self.title_label = Label(title, "subtitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.hint_label = Label(hint, "muted", wrap=True)
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.browse_button = SecondaryButton("Add videos…", "video")
        self.folder_button = SecondaryButton("Add folder…", "folder")
        self.browse_button.clicked.connect(self.browse_requested)
        self.folder_button.clicked.connect(self.folder_requested)
        for child in (self._badge, self.title_label, self.hint_label):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        buttons = QHBoxLayout()
        buttons.setSpacing(Spacing.SM)
        buttons.addStretch(1)
        buttons.addWidget(self.browse_button)
        buttons.addWidget(self.folder_button)
        buttons.addStretch(1)
        self._layout_box = QVBoxLayout(self)
        self._layout_box.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        self._layout_box.setSpacing(Spacing.SM)
        self._layout_box.addStretch(1)
        self._layout_box.addWidget(self._badge, 0, Qt.AlignmentFlag.AlignHCenter)
        self._layout_box.addWidget(self.title_label)
        self._layout_box.addWidget(self.hint_label)
        self._layout_box.addSpacing(Spacing.XS)
        self._layout_box.addLayout(buttons)
        self._layout_box.addStretch(1)

    # -- state ------------------------------------------------------------
    def is_dragging(self) -> bool:
        """True while a droppable item is held over the zone."""
        return self._dragging

    def recursive(self) -> bool:
        return self._recursive

    def set_recursive(self, recursive: bool) -> None:
        self._recursive = bool(recursive)

    def set_compact(self, compact: bool) -> None:
        """A short strip (for when videos are already listed)."""
        self._compact = compact
        self._badge.setVisible(not compact)
        self.hint_label.setVisible(not compact)
        margin = Spacing.SM if compact else Spacing.LG
        self._layout_box.setContentsMargins(Spacing.LG, margin, Spacing.LG, margin)

    # -- drag and drop ----------------------------------------------------
    @staticmethod
    def _has_local_files(mime) -> bool:
        return mime.hasUrls() and any(url.isLocalFile() for url in mime.urls())

    def _set_dragging(self, dragging: bool) -> None:
        if dragging != self._dragging:
            self._dragging = dragging
            self.update()

    def dragEnterEvent(self, event) -> None:
        if self._has_local_files(event.mimeData()):
            event.acceptProposedAction()
            self._set_dragging(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if self._has_local_files(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._set_dragging(False)
        event.accept()

    def dropEvent(self, event) -> None:
        self._set_dragging(False)
        videos, ignored = paths_from_mime(event.mimeData(), self._recursive)
        if not videos and not ignored:
            event.ignore()
            return
        event.acceptProposedAction()
        if ignored:
            self.paths_ignored.emit(ignored)
        if videos:
            self.paths_dropped.emit(videos)

    # -- click / keyboard / paint ----------------------------------------
    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.pos()):
            self.browse_requested.emit()

    def enterEvent(self, event) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover = False
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.browse_requested.emit()
        else:
            super().keyPressEvent(event)

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.update()

    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        if self._dragging:
            fill, border, width = palette.tint("primary", 0.12), palette.primary, 2.5
        elif self._hover:
            fill, border, width = palette.tint("primary", 0.05), palette.primary, 1.5
        else:
            fill, border, width = palette.surface, palette.border_strong, 1.5
        _rounded(painter, rect, 14, fill)
        pen = QPen(QColor(border), width, Qt.PenStyle.DashLine)
        pen.setDashPattern([5.0, 4.0])
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect.adjusted(width / 2, width / 2, -width / 2, -width / 2),
                                14, 14)
        if self.hasFocus():
            _focus_ring(painter, rect.adjusted(-2, -2, 2, 2), 14, palette)


# ----------------------------------------------------------------------
# Paradigm cards
# ----------------------------------------------------------------------
PARADIGM_INFO: Dict[str, Tuple[str, str]] = {
    "Y-Maze": ("maze-y", "Spontaneous alternation: arm entries and alternation %"),
    "Elevated Plus Maze": ("maze-plus", "Anxiety: time and entries in open vs closed arms"),
    "Freestyle / Open Field": ("maze-field",
                               "Locomotion, zones, center vs periphery (thigmotaxis)"),
    "Tail Suspension Test": ("maze-tst",
                             "Immobility, passive swing and active escape per mouse"),
}


class ParadigmCard(_Themed, QAbstractButton):
    """A selectable card for one paradigm: icon, name, one-line description."""

    navigate = pyqtSignal(int)      # arrow key: -1 previous, +1 next

    def __init__(self, name: str, description: str = "", icon_name: str = "maze-field",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._name = name
        self._description = description
        self._icon_name = icon_name
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumSize(120, 124)    # the picker reflows long before this
        self.setAccessibleName(name)
        self.setAccessibleDescription(description)
        self.setToolTip(description)
        self._hover = False

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def sizeHint(self) -> QSize:
        return QSize(260, 124)

    def enterEvent(self, event) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover = False
        self.update()

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.update()

    def keyPressEvent(self, event) -> None:
        step = {Qt.Key.Key_Left: -1, Qt.Key.Key_Up: -1,
                Qt.Key.Key_Right: 1, Qt.Key.Key_Down: 1}.get(event.key())
        if step:
            self.navigate.emit(step)
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.click()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event) -> None:
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        selected = self.isChecked()
        if selected:
            _rounded(painter, rect, Radius.LG, palette.tint("primary", 0.08),
                     palette.primary, 2.0)
        else:
            _rounded(painter, rect, Radius.LG,
                     palette.surface_alt if self._hover else palette.surface,
                     palette.border_strong if self._hover else palette.border, 1.0)
        if self.hasFocus():
            _focus_ring(painter, rect, Radius.LG, palette)
        pad = 16.0
        tile = QRectF(rect.left() + pad, rect.top() + pad, 40, 40)
        _rounded(painter, tile, 11, palette.tint("primary", 0.12))
        _draw_icon(painter, self, self._icon_name, palette.primary, tile.adjusted(9, 9, -9, -9))
        title_font = _font(self.font(), TypeScale.SUBTITLE, True)
        text_left = tile.right() + 12
        right_limit = rect.right() - pad - (26 if selected else 0)
        painter.setFont(title_font)
        painter.setPen(QColor(palette.primary if selected else palette.text))
        painter.drawText(
            QRectF(text_left, tile.top(), right_limit - text_left, tile.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            _elide(self._name, title_font, right_limit - text_left),
        )
        body_font = _font(self.font(), TypeScale.BODY)
        painter.setFont(body_font)
        painter.setPen(QColor(palette.text_muted))
        painter.drawText(
            QRectF(rect.left() + pad, tile.bottom() + 10, rect.width() - 2 * pad,
                   rect.bottom() - tile.bottom() - 10 - 10),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            | int(Qt.TextFlag.TextWordWrap),
            self._description,
        )
        if selected:
            badge = QRectF(rect.right() - pad - 22, tile.top() + 9, 22, 22)
            _rounded(painter, badge, 11, palette.primary)
            _draw_icon(painter, self, "check", palette.primary_text, badge.adjusted(4, 4, -4, -4))


class ParadigmPicker(_Themed, QWidget):
    """Cards for the registered paradigms. ``paradigm_chosen(name)`` is
    emitted when the user picks one (click, Enter, Space or arrow keys);
    ``set_selected`` changes the choice silently unless ``emit=True``."""

    paradigm_chosen = pyqtSignal(str)

    MIN_CARD_WIDTH = 230

    def __init__(self, names: Optional[Sequence[str]] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        if names is None:
            from mazes import AVAILABLE_MAZES
            names = list(AVAILABLE_MAZES)
        self._cards: Dict[str, ParadigmCard] = {}
        self._selected: Optional[str] = None
        self._columns = 0
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(Spacing.SM)
        for name in names:
            icon_name, description = PARADIGM_INFO.get(name, ("maze-field", "Custom paradigm"))
            card = ParadigmCard(name, description, icon_name)
            card.clicked.connect(lambda _checked=False, n=name: self._choose(n))
            card.navigate.connect(lambda step, n=name: self._navigate(n, step))
            self._cards[name] = card
        self._relayout(max(1, len(self._cards)))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def names(self) -> List[str]:
        return list(self._cards)

    def card(self, name: str) -> ParadigmCard:
        return self._cards[name]

    def selected(self) -> Optional[str]:
        return self._selected

    def columns(self) -> int:
        return self._columns

    def set_selected(self, name: Optional[str], emit: bool = False) -> None:
        if name is not None and name not in self._cards:
            raise KeyError(name)
        self._selected = name
        for card_name, card in self._cards.items():
            card.setChecked(card_name == name)
        if emit and name is not None:
            self.paradigm_chosen.emit(name)

    def _choose(self, name: str) -> None:
        changed = name != self._selected
        self.set_selected(name)
        if changed:
            self.paradigm_chosen.emit(name)

    def _navigate(self, name: str, step: int) -> None:
        names = self.names()
        target = names.index(name) + step
        if 0 <= target < len(names):
            card = self._cards[names[target]]
            card.setFocus(Qt.FocusReason.TabFocusReason)
            self._choose(names[target])

    def _relayout(self, columns: int) -> None:
        columns = max(1, min(columns, len(self._cards) or 1))
        if columns == self._columns:
            return
        self._columns = columns
        for card in self._cards.values():
            self._grid.removeWidget(card)
        for index, card in enumerate(self._cards.values()):
            self._grid.addWidget(card, index // columns, index % columns)
        for column in range(columns):
            self._grid.setColumnStretch(column, 1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        gap = self._grid.spacing()
        self._relayout(max(1, (self.width() + gap) // (self.MIN_CARD_WIDTH + gap)))


# ----------------------------------------------------------------------
# Filmstrip
# ----------------------------------------------------------------------
@dataclass
class FilmstripItem:
    """One video in a ``Filmstrip``."""
    key: str
    path: Path
    name: str = ""
    badge_text: str = ""
    badge_kind: str = "neutral"
    tooltip: str = ""

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.name:
            self.name = self.path.name


KeyRole = Qt.ItemDataRole.UserRole


class _FilmModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: List[FilmstripItem] = []

    def rowCount(self, parent: Optional[QModelIndex] = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.items)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.items):
            return None
        item = self.items[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return item.name
        if role == Qt.ItemDataRole.ToolTipRole:
            return item.tooltip or str(item.path)
        if role == KeyRole:
            return item.key
        return None

    def set_items(self, items: Sequence[FilmstripItem]) -> None:
        self.beginResetModel()
        self.items = list(items)
        self.endResetModel()

    def row_of(self, key: str) -> int:
        for row, item in enumerate(self.items):
            if item.key == key:
                return row
        return -1

    def update_badge(self, key: str, text: str, kind: str) -> bool:
        row = self.row_of(key)
        if row < 0:
            return False
        self.items[row].badge_text = text
        self.items[row].badge_kind = kind
        index = self.index(row)
        self.dataChanged.emit(index, index)
        return True


THUMB_CARD = QSize(148, 138)
THUMB_IMAGE = QSize(148, 83)       # 16:9
ROW_HEIGHT_FULL = 72
ROW_HEIGHT_COMPACT = 32


class _FilmDelegate(QStyledItemDelegate):
    """Paints thumbnails (cards or rows) and compact list rows."""

    def __init__(self, strip: "Filmstrip", view: QListView):
        super().__init__(view)
        self._strip = strip
        self._view = view
        self._pixmaps: Dict[Tuple[str, int, int, int], QPixmap] = {}

    # -- geometry ---------------------------------------------------------
    def sizeHint(self, option, index) -> QSize:
        strip = self._strip
        if strip.is_compact():
            return QSize(max(120, self._view.viewport().width()), ROW_HEIGHT_COMPACT)
        if strip.orientation() == Qt.Orientation.Horizontal:
            return THUMB_CARD
        return QSize(max(160, self._view.viewport().width()), ROW_HEIGHT_FULL)

    # -- thumbnails -------------------------------------------------------
    def _thumbnail(self, item: FilmstripItem, size: QSize) -> Tuple[Optional[QPixmap], bool]:
        """``(pixmap or None, failed)``; asks the loader when it is not ready."""
        loader = self._strip.thumbnail_loader
        preview = loader.get(item.path)
        if preview is None:
            loader.request(item.path)
            return None, False
        if not preview.ok:
            return None, True
        ratio = self._view.devicePixelRatioF()
        key = (str(item.path), size.width(), size.height(), round(ratio * 100))
        pixmap = self._pixmaps.get(key)
        if pixmap is None:
            if len(self._pixmaps) > 256:
                self._pixmaps.clear()
            scaled = preview.image.scaled(
                int(size.width() * ratio), int(size.height() * ratio),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            pixmap = QPixmap.fromImage(scaled)
            pixmap.setDevicePixelRatio(ratio)
            self._pixmaps[key] = pixmap
        return pixmap, False

    def _paint_thumb(self, painter: QPainter, rect: QRectF, item: FilmstripItem,
                     palette: Palette) -> None:
        pixmap, failed = self._thumbnail(item, rect.size().toSize())
        painter.save()
        clip = QPainterPath()
        clip.addRoundedRect(rect, 8, 8)
        painter.setClipPath(clip)
        if pixmap is not None:
            width = pixmap.width() / pixmap.devicePixelRatio()
            height = pixmap.height() / pixmap.devicePixelRatio()
            painter.drawPixmap(
                QPointF(rect.center().x() - width / 2, rect.center().y() - height / 2), pixmap)
        else:
            painter.fillRect(rect, QColor(palette.surface_alt))
            _draw_icon(painter, self._view, "warning" if failed else "film",
                       palette.text_disabled if not failed else palette.warning,
                       QRectF(rect.center().x() - 12, rect.center().y() - 12, 24, 24))
        painter.restore()

    # -- painting ---------------------------------------------------------
    def paint(self, painter: QPainter, option, index) -> None:
        item: FilmstripItem = self._strip.item_at(index.row())
        palette = current_palette()
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(option.rect).adjusted(2, 2, -2, -2)
        if self._strip.is_compact():
            self._paint_compact(painter, rect, item, palette, selected, hover)
        elif self._strip.orientation() == Qt.Orientation.Horizontal:
            self._paint_card(painter, rect, item, palette, selected, hover)
        else:
            self._paint_row(painter, rect, item, palette, selected, hover)
        painter.restore()

    def _paint_card(self, painter, rect, item, palette, selected, hover) -> None:
        _rounded(painter, rect, 10,
                 palette.tint("primary", 0.10) if selected
                 else palette.surface_alt if hover else palette.surface,
                 palette.primary if selected else palette.border, 2.0 if selected else 1.0)
        thumb = QRectF(rect.left() + 6, rect.top() + 6, rect.width() - 12,
                       (rect.width() - 12) * 9 / 16)
        self._paint_thumb(painter, thumb, item, palette)
        font = _font(self._view.font(), 12, selected)
        painter.setFont(font)
        painter.setPen(QColor(palette.primary if selected else palette.text))
        name_top = thumb.bottom() + 6
        painter.drawText(QRectF(thumb.left(), name_top, thumb.width(), 16),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         _elide(item.name, font, thumb.width()))
        self._paint_badge(painter, QRectF(thumb.left(), name_top + 20, thumb.width(), 22), item)

    def _paint_row(self, painter, rect, item, palette, selected, hover) -> None:
        _rounded(painter, rect, 10,
                 palette.tint("primary", 0.10) if selected
                 else palette.surface_alt if hover else palette.surface,
                 palette.primary if selected else palette.border, 2.0 if selected else 1.0)
        thumb = QRectF(rect.left() + 6, rect.top() + 6, 80, rect.height() - 12)
        self._paint_thumb(painter, thumb, item, palette)
        left = thumb.right() + 10
        font = _font(self._view.font(), 12, True)
        painter.setFont(font)
        painter.setPen(QColor(palette.primary if selected else palette.text))
        painter.drawText(QRectF(left, rect.top() + 8, rect.right() - left - 8, 18),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         _elide(item.name, font, rect.right() - left - 8))
        self._paint_badge(painter, QRectF(left, rect.top() + 30, rect.right() - left - 8, 22), item)

    def _paint_compact(self, painter, rect, item, palette, selected, hover) -> None:
        if selected:
            _rounded(painter, rect, 6, palette.selection)
        elif hover:
            _rounded(painter, rect, 6, palette.surface_alt)
        _draw_icon(painter, self._view, "video",
                   palette.primary if selected else palette.text_muted,
                   QRectF(rect.left() + 8, rect.center().y() - 8, 16, 16))
        font = _font(self._view.font(), TypeScale.BODY, selected)
        right = rect.right() - 8
        if item.badge_text:
            chip_font = _font(self._view.font(), TypeScale.CAPTION, True)
            chip = _chip_size(item.badge_text, chip_font,
                              item.badge_kind in _CHIP_ICONS or item.badge_kind == "running")
            chip_rect = QRectF(right - chip.width(), rect.center().y() - chip.height() / 2,
                               chip.width(), chip.height())
            _paint_chip(painter, self._view, chip_rect, item.badge_text, item.badge_kind, chip_font)
            right = chip_rect.left() - 8
        painter.setFont(font)
        painter.setPen(QColor(palette.text))
        painter.drawText(QRectF(rect.left() + 32, rect.top(), right - rect.left() - 32, rect.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         _elide(item.name, font, right - rect.left() - 32))

    def _paint_badge(self, painter, area: QRectF, item: FilmstripItem) -> None:
        if not item.badge_text:
            return
        font = _font(self._view.font(), TypeScale.CAPTION, True)
        size = _chip_size(item.badge_text, font,
                          item.badge_kind in _CHIP_ICONS or item.badge_kind == "running")
        width = min(size.width(), area.width())
        _paint_chip(painter, self._view, QRectF(area.left(), area.top(), width, size.height()),
                    item.badge_text, item.badge_kind, font)


class Filmstrip(_Themed, QWidget):
    """A strip of video thumbnails (horizontal or vertical) that can collapse
    into a compact list. Thumbnails are read lazily, only for the items on
    screen, by a ``ThumbnailLoader`` (pass a shared one, or the strip makes
    its own).

    ``set_items([FilmstripItem(key, path, badge_text="Done", badge_kind="ok")])``,
    ``select(key)``, ``set_badge(key, text, kind)``. ``video_selected(key)``
    is emitted when the user picks a video (not by ``select``).
    """

    video_selected = pyqtSignal(str)
    compact_changed = pyqtSignal(bool)

    def __init__(self, orientation: Qt.Orientation = Qt.Orientation.Horizontal,
                 loader: Optional[ThumbnailLoader] = None, title: str = "Videos",
                 show_header: bool = True, settings=None, settings_key: str = "",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._orientation = orientation
        self._settings = settings
        self._key = f"compact/{settings_key}" if settings is not None and settings_key else ""
        self._compact = bool(self._key and settings.ui_flag(self._key, default=False))
        self.thumbnail_loader = loader if loader is not None else ThumbnailLoader(self)
        self._title = title
        self._syncing = False
        self._model = _FilmModel(self)
        self.view = QListView(self)
        self.view.setProperty("flat", True)
        self.view.setModel(self._model)
        self.view.setItemDelegate(_FilmDelegate(self, self.view))
        self.view.setMouseTracking(True)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.view.setViewMode(QListView.ViewMode.ListMode)
        self.view.setWrapping(False)
        self.view.setSpacing(2)
        self.view.setFrameShape(QFrame.Shape.NoFrame)
        self.view.setUniformItemSizes(False)
        self.view.selectionModel().currentChanged.connect(self._current_changed)
        self.thumbnail_loader.loaded.connect(self._thumbnail_ready)
        self.title_label = Label(title, "subtitle")
        self.count_label = Label("", "muted")
        self.toggle_button = GhostButton("", "list")
        self.toggle_button.setFixedSize(34, 34)
        self.toggle_button.clicked.connect(self.toggle_compact)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(Spacing.XS)
        header.addWidget(self.title_label)
        header.addWidget(self.count_label)
        header.addStretch(1)
        header.addWidget(self.toggle_button)
        self._header = QWidget(self)
        self._header.setLayout(header)
        self._header.setVisible(show_header)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(Spacing.XS)
        column.addWidget(self._header)
        column.addWidget(self.view, 1)
        self._apply_mode()

    # -- content ----------------------------------------------------------
    def set_items(self, items: Sequence[FilmstripItem]) -> None:
        previous = self.selected_key()
        self._syncing = True
        try:
            self._model.set_items(items)
            if previous is not None:
                self._select_row(self._model.row_of(previous))
        finally:
            self._syncing = False
        self.count_label.setText(f"({len(items)})" if items else "")

    def items(self) -> List[FilmstripItem]:
        return list(self._model.items)

    def item_at(self, row: int) -> FilmstripItem:
        return self._model.items[row]

    def keys(self) -> List[str]:
        return [item.key for item in self._model.items]

    def set_badge(self, key: str, text: str, kind: str = "neutral") -> bool:
        if kind not in CHIP_KINDS:
            raise ValueError(f"Unknown chip kind {kind!r}")
        return self._model.update_badge(key, text, kind)

    # -- selection --------------------------------------------------------
    def selected_key(self) -> Optional[str]:
        index = self.view.currentIndex()
        return index.data(KeyRole) if index.isValid() else None

    def select(self, key: Optional[str], emit: bool = False) -> None:
        """Highlight a video (``None`` clears). Silent unless ``emit``."""
        row = self._model.row_of(key) if key is not None else -1
        self._syncing = not emit
        try:
            self._select_row(row)
        finally:
            self._syncing = False

    def _select_row(self, row: int) -> None:
        if row < 0:
            self.view.selectionModel().clear()
            self.view.setCurrentIndex(QModelIndex())
            return
        index = self._model.index(row)
        self.view.setCurrentIndex(index)
        self.view.selectionModel().select(
            index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        self.view.scrollTo(index)

    def _current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        if self._syncing or not current.isValid():
            return
        self.video_selected.emit(current.data(KeyRole))

    def _thumbnail_ready(self, _path: str) -> None:
        self.view.viewport().update()

    # -- layout modes -----------------------------------------------------
    def orientation(self) -> Qt.Orientation:
        return self._orientation

    def is_compact(self) -> bool:
        return self._compact

    def set_compact(self, compact: bool) -> None:
        if compact == self._compact:
            return
        self._compact = compact
        if self._key:
            self._settings.set_ui_flag(self._key, compact)
        self._apply_mode()
        self.compact_changed.emit(compact)

    def toggle_compact(self) -> None:
        self.set_compact(not self._compact)

    def _apply_mode(self) -> None:
        horizontal_strip = self._orientation == Qt.Orientation.Horizontal and not self._compact
        self.view.setFlow(QListView.Flow.LeftToRight if horizontal_strip
                          else QListView.Flow.TopToBottom)
        if self._orientation == Qt.Orientation.Horizontal:
            self.view.setFixedHeight(THUMB_CARD.height() + 22 if not self._compact else 128)
            self.view.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff if horizontal_strip
                else Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.view.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded if horizontal_strip
                else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        else:
            self.view.setMinimumHeight(0)
            self.view.setMaximumHeight(16777215)
            self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.toggle_button.set_icon_name("film" if self._compact else "list")
        self.toggle_button.setToolTip("Show thumbnails" if self._compact else "Show as a list")
        self.toggle_button.setAccessibleName(self.toggle_button.toolTip())
        self.view.doItemsLayout()
        self.view.viewport().update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not (self._orientation == Qt.Orientation.Horizontal and not self._compact):
            self.view.doItemsLayout()


