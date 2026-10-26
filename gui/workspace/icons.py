"""
gui/workspace/icons.py
Clean line icons drawn as inline SVG and rendered with ``QSvgRenderer``.

Every icon lives on a 24x24 grid and is stroked (never filled) with round
caps, so one string serves every size, colour and screen density. ``icon()``
returns a ``QIcon`` backed by an icon engine that renders on demand at the
size Qt asks for, so the icon stays sharp at any scale factor and a disabled
button greys its icon without any extra work.

``color`` may be a ``QColor``, a ``#rrggbb`` string or the *name of a palette
token* (``"primary"``, ``"text_muted"`` ...). A token is looked up when the
icon is painted, so an icon created before a light/dark switch follows the
theme without being rebuilt.
"""
from typing import Dict, Optional, Tuple, Union

from PyQt6.QtCore import QByteArray, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QIconEngine, QImage, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

DEFAULT_ICON_SIZE = 20
STROKE_WIDTH = 1.75

# Inner markup of each icon (the <svg> wrapper and stroke style are added).
_ICONS: Dict[str, str] = {
    "video": '<rect x="3" y="6" width="13" height="12" rx="2"/>'
             '<path d="M16 10.5l5-3v9l-5-3z"/>',
    "film": '<rect x="3" y="4" width="18" height="16" rx="2"/>'
            '<path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5'
              'a2 2 0 0 1-2-2z"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "trash": '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2'
             'l1-12M9 7V4h6v3"/>',
    "play": '<path d="M7 5l12 7-12 7z"/>',
    "pause": '<path d="M8.5 5v14M15.5 5v14"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "check-circle": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l3 3 5-6"/>',
    "warning": '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17.2v.1"/>',
    "error": '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.8v.1"/>',
    "help": '<circle cx="12" cy="12" r="9"/>'
            '<path d="M9.5 9.6a2.6 2.6 0 0 1 5 1c0 1.7-2.5 2-2.5 3.6M12 17.2v.1"/>',
    "chevron-left": '<path d="M14.5 6l-6 6 6 6"/>',
    "chevron-right": '<path d="M9.5 6l6 6-6 6"/>',
    "chevron-down": '<path d="M6 9.5l6 6 6-6"/>',
    "chevron-up": '<path d="M6 14.5l6-6 6 6"/>',
    "arrow-left": '<path d="M19 12H5M11 6l-6 6 6 6"/>',
    "arrow-right": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    "close": '<path d="M6 6l12 12M18 6L6 18"/>',
    "settings": '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/>'
                '<circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
    "zones": '<path d="M5 8l7-4 7 5-2 9-9 2z"/>'
             '<circle cx="5" cy="8" r=".6"/><circle cx="12" cy="4" r=".6"/>'
             '<circle cx="19" cy="9" r=".6"/><circle cx="17" cy="18" r=".6"/>'
             '<circle cx="8" cy="20" r=".6"/>',
    "target": '<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="2.5"/>'
              '<path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "chart": '<path d="M4 4v16h16"/><path d="M8 15l3-4 3 2 5-6"/>',
    "list": '<path d="M9 6h11M9 12h11M9 18h11M4.5 6h.01M4.5 12h.01M4.5 18h.01"/>',
    "search": '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>',
    "eye": '<path d="M2 12s3.7-7 10-7 10 7 10 7-3.7 7-10 7S2 12 2 12z"/>'
           '<circle cx="12" cy="12" r="3"/>',
    "undo": '<path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
    "redo": '<path d="M15 14l5-5-5-5"/><path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13"/>',
    "zoom-in": '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5M8.5 11h5M11 8.5v5"/>',
    "zoom-out": '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5M8.5 11h5"/>',
    "fit": '<path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4'
           'M9 20H5a1 1 0 0 1-1-1v-4"/>',
    "align": '<path d="M12 3v18M3 12h18M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3"/>',
    "reset": '<path d="M4 12a8 8 0 1 0 2.4-5.7"/><path d="M4 4v4.5h4.5"/>',
    "ruler": '<rect x="2.5" y="8" width="19" height="8" rx="1.5"/>'
             '<path d="M6.5 8v3M10.5 8v4M14.5 8v3M18.5 8v4"/>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    "image": '<rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="10" r="1.5"/>'
             '<path d="M21 16l-5-5-8 8"/>',
    # Paradigms.
    "maze-y": '<path d="M12 21v-8M12 13L5 4M12 13l7-9"/>',
    "maze-plus": '<path d="M12 3v18M3 12h18"/><rect x="9" y="9" width="6" height="6" rx="1"/>',
    "maze-field": '<rect x="3.5" y="3.5" width="17" height="17" rx="2.5"/>'
                  '<circle cx="12" cy="12" r="5"/>',
    "maze-tst": '<path d="M5 4h14M12 4v6"/><ellipse cx="12" cy="15.5" rx="3.5" ry="5"/>',
}

ICON_NAMES: Tuple[str, ...] = tuple(_ICONS)

ColorLike = Union[QColor, str, None]


def _resolve_color(color: ColorLike, mode_disabled: bool = False) -> QColor:
    """A QColor from a QColor, ``#hex``, palette token name, or None (text)."""
    from gui.workspace.theme import current_palette    # late: avoids a cycle
    palette = current_palette()
    if isinstance(color, QColor):
        resolved = QColor(color)
    elif color is None:
        resolved = QColor(palette.text)
    elif color.startswith("#") or not hasattr(palette, color):
        resolved = QColor(color)
        if not resolved.isValid():
            resolved = QColor(palette.text)
    else:
        resolved = QColor(getattr(palette, color))
    if mode_disabled:
        resolved.setAlphaF(resolved.alphaF() * 0.4)
    return resolved


def icon_svg(name: str, color: str = "#000000", stroke: float = STROKE_WIDTH) -> str:
    """The complete SVG document of an icon (``color`` is a ``#rrggbb`` string)."""
    try:
        body = _ICONS[name]
    except KeyError:
        raise KeyError(f"Unknown icon {name!r}") from None
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round">{body}</svg>'
    )


def render_icon(name: str, color: QColor, size: QSize, device_ratio: float = 1.0) -> QPixmap:
    """Render an icon to a transparent pixmap of ``size`` logical pixels."""
    width = max(1, round(size.width() * device_ratio))
    height = max(1, round(size.height() * device_ratio))
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    # SVG colours cannot carry alpha portably, so draw opaque and apply it.
    opaque = QColor(color)
    alpha = opaque.alphaF()
    opaque.setAlpha(255)
    renderer = QSvgRenderer(QByteArray(icon_svg(name, opaque.name()).encode("utf-8")))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if alpha < 1.0:
        painter.setOpacity(alpha)
    renderer.render(painter, QRectF(0, 0, width, height))
    painter.end()
    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(device_ratio)
    return pixmap


class _SvgIconEngine(QIconEngine):
    """Renders one named icon at whatever size and mode Qt asks for."""

    def __init__(self, name: str, color: ColorLike, size: Optional[int]):
        super().__init__()
        self._name = name
        self._color = color
        self._size = size

    def clone(self) -> "_SvgIconEngine":
        return _SvgIconEngine(self._name, self._color, self._size)

    def actualSize(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QSize:
        if self._size:
            return QSize(self._size, self._size)
        side = min(size.width(), size.height()) if size.isValid() else DEFAULT_ICON_SIZE
        return QSize(side, side)

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        actual = self.actualSize(size, mode, state)
        disabled = mode == QIcon.Mode.Disabled
        return render_icon(self._name, _resolve_color(self._color, disabled), actual)

    def paint(self, painter: QPainter, rect: QRect, mode: QIcon.Mode,
              state: QIcon.State) -> None:
        ratio = 1.0
        device = painter.device()
        if device is not None:
            ratio = max(1.0, float(device.devicePixelRatioF()))
        disabled = mode == QIcon.Mode.Disabled
        pixmap = render_icon(
            self._name, _resolve_color(self._color, disabled), rect.size(), ratio
        )
        painter.drawPixmap(rect, pixmap)


def icon(name: str, color: ColorLike = None, size: Optional[int] = None) -> QIcon:
    """A ``QIcon`` for the named line icon.

    ``color``: QColor, ``#rrggbb`` or a palette token name (default: the
    theme's text colour). ``size``: a fixed logical size in px (default: the
    size the widget asks for).
    """
    if name not in _ICONS:
        raise KeyError(f"Unknown icon {name!r}")
    return QIcon(_SvgIconEngine(name, color, size))


def icon_pixmap(name: str, color: ColorLike = None, size: int = DEFAULT_ICON_SIZE,
                device_ratio: float = 1.0) -> QPixmap:
    """The icon as a plain pixmap (for painting and labels)."""
    return render_icon(
        name, _resolve_color(color), QSize(size, size), device_ratio
    )
