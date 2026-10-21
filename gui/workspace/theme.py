"""
gui/workspace/theme.py
The workspace design system: colour tokens (light and dark), spacing, radii,
the type scale, one application-wide Qt style sheet, and the machinery that
applies them and follows the operating system's colour scheme.

How to use it
-------------
* ``apply_theme(app, mode="auto")`` once at start-up (``gui_main`` does it for
  the workspace window only; the classic window keeps the platform look).
  ``mode`` is ``"auto"`` (follow the OS), ``"light"`` or ``"dark"``. While the
  mode is ``"auto"`` the theme switches live when the OS scheme changes.
* ``current_palette()`` is the palette in force; custom-painted widgets read
  colours from it at paint time and repaint on ``QEvent.PaletteChange``,
  which ``apply_theme`` triggers for every widget.
* Style a widget through dynamic properties instead of per-widget style
  sheets:

  - ``QPushButton`` ``class``: ``"primary"``, ``"secondary"`` (the default
    look), ``"ghost"`` or ``"danger"`` (see ``set_class``);
  - ``QLabel`` ``role``: ``"display"``, ``"title"``, ``"subtitle"``,
    ``"body"``, ``"caption"`` or ``"muted"`` (see ``set_role``);
  - any input ``invalid="true"`` draws the error border.

* ``icon(name, color=None, size=None)`` gives a ``QIcon`` from the line-icon
  set in :mod:`gui.workspace.icons`.
* ``style_table_view(view)`` applies the comfortable 32 px, zebra, grid-free
  table look to a ``QTableView`` (the style sheet colours it; row height and
  the header are widget settings).
"""
import hashlib
import logging
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from string import Template
from typing import Dict, Optional

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QPalette
from PyQt6.QtWidgets import QAbstractItemView, QApplication, QTableView, QWidget

from gui.workspace.icons import ICON_NAMES, icon, icon_pixmap, icon_svg, render_icon

logger = logging.getLogger(__name__)

__all__ = [
    "ICON_NAMES", "DARK", "LIGHT", "Palette", "Radius", "Spacing", "ThemeController",
    "TypeScale", "apply_theme", "contrast_ratio", "current_palette", "font_family",
    "get_controller", "icon", "icon_pixmap", "icon_svg", "mix", "palette_for",
    "render_icon", "reset_theme", "set_class", "set_role", "stylesheet",
    "style_table_view",
]

MODES = ("auto", "light", "dark")


# ----------------------------------------------------------------------
# Scales
# ----------------------------------------------------------------------
class Spacing:
    """Spacing scale in px: use these for margins and gaps."""
    XXS = 4
    XS = 8
    SM = 12
    MD = 16
    LG = 24
    XL = 32
    XXL = 48
    SCALE = (4, 8, 12, 16, 24, 32, 48)


class Radius:
    """Corner radii in px."""
    SM = 4         # checkboxes, small chips
    MD = 6         # buttons, inputs
    LG = 10        # cards, panels
    PILL = 999


class TypeScale:
    """Font sizes in px (and which are semibold)."""
    DISPLAY = 22
    TITLE = 18
    SUBTITLE = 14   # semibold
    BODY = 13
    CAPTION = 11
    SIZES = {"display": 22, "title": 18, "subtitle": 14, "body": 13, "caption": 11}
    SEMIBOLD = frozenset({"display", "title", "subtitle"})


# ----------------------------------------------------------------------
# Colour tokens
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Palette:
    """One colour scheme. All values are ``#rrggbb`` strings."""
    name: str
    background: str        # window background
    surface: str           # cards, inputs, tables
    surface_alt: str       # hover, zebra rows, quiet fills
    border: str            # decorative separators
    border_strong: str     # input and control outlines (>= 3:1 on surface)
    text: str
    text_muted: str
    text_disabled: str
    primary: str
    primary_hover: str
    primary_text: str      # text and icons on a primary fill
    success: str
    warning: str
    danger: str
    info: str
    focus_ring: str
    selection: str         # selected rows and menu items (text stays ``text``)

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"

    def tint(self, color: str, amount: float = 0.10, over: Optional[str] = None) -> str:
        """``color`` faintly mixed into a base (default: ``surface``).

        ``color`` may be a token name (``"danger"``) or a hex colour. The
        status colours keep at least 4.5:1 against their own tint.
        """
        base = over if over is not None else self.surface
        return mix(getattr(self, color, color), getattr(self, base, base), amount)

    def kind_colors(self, kind: str) -> "tuple[str, str]":
        """``(foreground, soft background)`` for a status kind.

        Kinds: ok/success, warning, error/danger, info, running, neutral.
        """
        key = {"ok": "success", "success": "success", "warning": "warning",
               "error": "danger", "danger": "danger", "info": "info",
               "tip": "info", "running": "primary"}.get(kind)
        if key is None:
            return self.text_muted, self.surface_alt
        return getattr(self, key), self.tint(key)


LIGHT = Palette(
    name="light",
    background="#F3F6F7", surface="#FFFFFF", surface_alt="#EDF1F3",
    border="#D8DFE3", border_strong="#8A9AA2",
    text="#18232A", text_muted="#566770", text_disabled="#7C8B93",
    primary="#0D6B63", primary_hover="#0A564F", primary_text="#FFFFFF",
    success="#13692F", warning="#855000", danger="#B42318", info="#1D5FBF",
    focus_ring="#2563EB", selection="#CDE7E3",
)

DARK = Palette(
    name="dark",
    background="#10181C", surface="#182126", surface_alt="#1F2A30",
    border="#2C3A41", border_strong="#6F858F",
    text="#E7EEF0", text_muted="#A3B4BB", text_disabled="#6C7D84",
    primary="#2DD4BF", primary_hover="#5EEAD4", primary_text="#06302B",
    success="#4ADE80", warning="#FBBF24", danger="#FF8A80", info="#8DB8FF",
    focus_ring="#7DB4FF", selection="#1F4A47",
)

_COLOR_FIELDS = tuple(f.name for f in fields(Palette) if f.name != "name")


def mix(foreground: str, background: str, amount: float) -> str:
    """``foreground`` over ``background`` at ``amount`` (0..1) opacity."""
    fg, bg = QColor(foreground), QColor(background)
    channels = [
        round(f * amount + b * (1 - amount))
        for f, b in ((fg.red(), bg.red()), (fg.green(), bg.green()),
                     (fg.blue(), bg.blue()))
    ]
    return "#{:02X}{:02X}{:02X}".format(*channels)


def _relative_luminance(color: str) -> float:
    qcolor = QColor(color)

    def linear(value: int) -> float:
        channel = value / 255
        return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

    return (0.2126 * linear(qcolor.red()) + 0.7152 * linear(qcolor.green())
            + 0.0722 * linear(qcolor.blue()))


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 2 contrast ratio of two colours (1 to 21)."""
    first, second = _relative_luminance(foreground), _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def palette_for(mode: str, scheme: Optional[Qt.ColorScheme] = None) -> Palette:
    """The palette of ``mode`` (``"auto"`` consults ``scheme``, light if unknown)."""
    if mode == "dark":
        return DARK
    if mode == "light":
        return LIGHT
    return DARK if scheme == Qt.ColorScheme.Dark else LIGHT


def font_family() -> str:
    """The platform's default UI font family (Segoe UI on Windows)."""
    family = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
    return family or "Segoe UI"


# ----------------------------------------------------------------------
# Generated image assets (arrows and check marks cannot be drawn by QSS)
# ----------------------------------------------------------------------
def _asset_dir(palette: Palette) -> Path:
    digest = hashlib.sha1(repr(palette).encode("utf-8")).hexdigest()[:10]
    return Path(tempfile.gettempdir()) / "BehavioralMazeAnalyzer-theme" / f"{palette.name}-{digest}"


def _asset_specs(palette: Palette) -> Dict[str, "tuple[str, str]"]:
    return {
        "chevron-down": ("chevron-down", palette.text_muted),
        "chevron-up": ("chevron-up", palette.text_muted),
        "chevron-down-disabled": ("chevron-down", palette.text_disabled),
        "chevron-up-disabled": ("chevron-up", palette.text_disabled),
        "check": ("check", palette.primary_text),
    }


def ensure_assets(palette: Palette) -> Path:
    """Write the small PNGs the style sheet refers to; returns their folder."""
    from PyQt6.QtCore import QSize
    folder = _asset_dir(palette)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for stem, (icon_name, color) in _asset_specs(palette).items():
            for suffix, ratio in (("", 1.0), ("@2x", 2.0)):
                target = folder / f"{stem}{suffix}.png"
                if not target.exists():
                    render_icon(icon_name, QColor(color), QSize(12, 12), ratio).save(
                        str(target), "PNG"
                    )
    except OSError:
        logger.warning("Could not write theme images to %s", folder, exc_info=True)
    return folder


# ----------------------------------------------------------------------
# Style sheet
# ----------------------------------------------------------------------
_QSS = Template(r"""
/* ---- base ------------------------------------------------------------ */
QMainWindow, QDialog { background: $background; }
QWidget { color: $text; }
QWidget:disabled { color: $text_disabled; }
QLabel { background: transparent; }
QLabel[role="display"] { font-size: 22px; font-weight: 600; }
QLabel[role="title"] { font-size: 18px; font-weight: 600; }
QLabel[role="subtitle"] { font-size: 14px; font-weight: 600; }
QLabel[role="body"] { font-size: 13px; }
QLabel[role="caption"] { font-size: 11px; color: $text_muted; }
QLabel[role="muted"] { color: $text_muted; }
QLabel[role="caption"]:disabled, QLabel[role="muted"]:disabled { color: $text_disabled; }
QFrame[frameShape="0"] { border: none; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QStackedWidget { background: transparent; }

/* ---- buttons --------------------------------------------------------- */
QPushButton, QToolButton {
    background: $surface; color: $text;
    border: 1px solid $border_strong; border-radius: 6px;
    padding: 6px 14px; min-height: 20px;
}
QPushButton:hover, QToolButton:hover { background: $surface_alt; }
QPushButton:pressed, QToolButton:pressed { background: $border; }
QPushButton:checked, QToolButton:checked {
    background: $primary_soft; border-color: $primary; color: $primary;
}
QPushButton:focus, QToolButton:focus {
    border: 2px solid $focus_ring; padding: 5px 13px;
}
QPushButton:disabled, QToolButton:disabled {
    background: $surface_alt; color: $text_disabled; border-color: $border;
}
QPushButton:default { border-color: $primary; }
QToolButton { padding: 4px 8px; border-color: transparent; background: transparent; }
QToolButton:focus { padding: 3px 7px; }
QToolButton::menu-indicator { image: none; width: 0px; }

QPushButton[class="primary"] {
    background: $primary; color: $primary_text; border: 1px solid $primary;
    font-weight: 600;
}
QPushButton[class="primary"]:hover { background: $primary_hover; border-color: $primary_hover; }
QPushButton[class="primary"]:pressed { background: $primary_hover; }
QPushButton[class="primary"]:focus { border: 2px solid $focus_ring; padding: 5px 13px; }
QPushButton[class="primary"]:disabled {
    background: $border; color: $text_disabled; border-color: $border;
}

QPushButton[class="secondary"] { }

QPushButton[class="ghost"] {
    background: transparent; border: 1px solid transparent; color: $text;
}
QPushButton[class="ghost"]:hover { background: $surface_alt; }
QPushButton[class="ghost"]:pressed { background: $border; }
QPushButton[class="ghost"]:focus { border: 2px solid $focus_ring; padding: 5px 13px; }
QPushButton[class="ghost"]:disabled { background: transparent; color: $text_disabled; }

QPushButton[class="danger"] {
    background: transparent; color: $danger; border: 1px solid $danger;
}
QPushButton[class="danger"]:hover { background: $danger_soft; }
QPushButton[class="danger"]:pressed { background: $danger_soft; }
QPushButton[class="danger"]:focus { border: 2px solid $focus_ring; padding: 5px 13px; }
QPushButton[class="danger"]:disabled {
    background: transparent; color: $text_disabled; border-color: $border;
}

/* ---- text inputs ----------------------------------------------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit,
QDateEdit, QTimeEdit, QDateTimeEdit {
    background: $surface; color: $text;
    border: 1px solid $border_strong; border-radius: 6px;
    padding: 5px 8px; min-height: 20px;
    selection-background-color: $selection; selection-color: $text;
}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover,
QPlainTextEdit:hover, QTextEdit:hover { border-color: $text_muted; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus, QDateEdit:focus, QTimeEdit:focus,
QDateTimeEdit:focus, QComboBox:on {
    border: 2px solid $focus_ring; padding: 4px 7px;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled,
QPlainTextEdit:disabled, QTextEdit:disabled {
    background: $surface_alt; color: $text_disabled; border-color: $border;
}
QLineEdit[invalid="true"], QSpinBox[invalid="true"], QDoubleSpinBox[invalid="true"],
QComboBox[invalid="true"], QPlainTextEdit[invalid="true"] { border: 2px solid $danger; padding: 4px 7px; }
QLineEdit[readOnly="true"] { background: $surface_alt; }

QComboBox { padding-right: 28px; }
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: center right;
    width: 24px; border: none; background: transparent;
}
QComboBox::down-arrow { image: url($asset_dir/chevron-down.png); width: 12px; height: 12px; }
QComboBox::down-arrow:disabled { image: url($asset_dir/chevron-down-disabled.png); }
QComboBox QAbstractItemView {
    background: $surface; border: 1px solid $border_strong; border-radius: 6px;
    padding: 4px; outline: 0; selection-background-color: $selection; selection-color: $text;
}
QComboBox QAbstractItemView::item { min-height: 28px; padding: 0 8px; border-radius: 4px; }

QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QDateTimeEdit { padding-right: 24px; }
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right;
    width: 20px; border: none; background: transparent; margin: 2px 2px 0 0;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 20px; border: none; background: transparent; margin: 0 2px 2px 0;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background: $surface_alt; border-radius: 4px;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url($asset_dir/chevron-up.png); width: 10px; height: 10px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url($asset_dir/chevron-down.png); width: 10px; height: 10px;
}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {
    image: url($asset_dir/chevron-up-disabled.png);
}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {
    image: url($asset_dir/chevron-down-disabled.png);
}

/* ---- check boxes and radio buttons ----------------------------------- */
QCheckBox, QRadioButton { spacing: 8px; background: transparent; }
QCheckBox::indicator, QRadioButton::indicator { width: 16px; height: 16px; }
QCheckBox::indicator {
    border: 1px solid $border_strong; border-radius: 4px; background: $surface;
}
QCheckBox::indicator:hover { border-color: $primary; }
QCheckBox::indicator:focus { border: 2px solid $focus_ring; }
QCheckBox::indicator:checked, QCheckBox::indicator:indeterminate {
    background: $primary; border-color: $primary; image: url($asset_dir/check.png);
}
QCheckBox::indicator:indeterminate { image: none; }
QCheckBox::indicator:disabled { background: $surface_alt; border-color: $border; }
QCheckBox::indicator:checked:disabled { background: $border; image: none; }
QRadioButton::indicator {
    width: 14px; height: 14px; border: 1px solid $border_strong; border-radius: 8px;
    background: $surface;
}
QRadioButton::indicator:hover { border-color: $primary; }
QRadioButton::indicator:focus { border: 2px solid $focus_ring; width: 13px; height: 13px; }
QRadioButton::indicator:checked {
    width: 6px; height: 6px; border: 5px solid $primary; border-radius: 8px;
    background: $surface;
}
QRadioButton::indicator:disabled { background: $surface_alt; border-color: $border; }

/* ---- item views ------------------------------------------------------ */
QTableView, QTreeView, QListView, QListWidget, QTableWidget, QTreeWidget {
    background: $surface; alternate-background-color: $surface_alt;
    border: 1px solid $border; border-radius: 8px; gridline-color: transparent;
    selection-background-color: $selection; selection-color: $text; outline: 0;
}
QTableView:focus, QTreeView:focus, QListView:focus { border-color: $focus_ring; }
QTableView::item, QTreeView::item, QListView::item {
    padding: 0 8px; border: none; min-height: 32px;
}
QTableView::item:hover, QTreeView::item:hover, QListView::item:hover { background: $row_hover; }
QTableView::item:selected, QTreeView::item:selected, QListView::item:selected {
    background: $selection; color: $text;
}
QTableView::item:disabled { color: $text_disabled; }
QHeaderView { background: transparent; border: none; }
QHeaderView::section {
    background: $surface; color: $text_muted; border: none;
    border-bottom: 1px solid $border; padding: 8px 8px; font-size: 12px; font-weight: 600;
}
QHeaderView::section:hover { background: $surface_alt; color: $text; }
QHeaderView::section:vertical { border-bottom: none; border-right: 1px solid $border; }
QTableCornerButton::section { background: $surface; border: none; }

/* ---- tabs ------------------------------------------------------------ */
QTabWidget::pane {
    border: 1px solid $border; border-radius: 8px; background: $surface; top: -1px;
}
QTabBar { background: transparent; }
QTabBar::tab {
    background: transparent; color: $text_muted; padding: 8px 16px;
    border: none; border-bottom: 2px solid transparent; margin-right: 4px;
}
QTabBar::tab:hover { color: $text; border-bottom-color: $border_strong; }
QTabBar::tab:selected { color: $primary; font-weight: 600; border-bottom: 2px solid $primary; }
QTabBar::tab:disabled { color: $text_disabled; }
QTabBar::tab:focus { background: $surface_alt; }

/* ---- scroll bars (thin) ---------------------------------------------- */
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:vertical { background: $border_strong; border-radius: 4px; min-height: 28px; margin: 1px; }
QScrollBar::handle:horizontal { background: $border_strong; border-radius: 4px; min-width: 28px; margin: 1px; }
QScrollBar::handle:hover { background: $text_muted; }
QScrollBar::handle:pressed { background: $text; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; background: none; border: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ---- sliders --------------------------------------------------------- */
QSlider::groove:horizontal { height: 4px; background: $border; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $primary; border-radius: 2px; }
QSlider::handle:horizontal {
    background: $primary; border: 2px solid $surface; width: 14px; height: 14px;
    margin: -7px 0; border-radius: 9px;
}
QSlider::handle:horizontal:hover { background: $primary_hover; }
QSlider::handle:horizontal:focus { border: 2px solid $focus_ring; }
QSlider::groove:vertical { width: 4px; background: $border; border-radius: 2px; }
QSlider::add-page:vertical { background: $primary; border-radius: 2px; }
QSlider::handle:vertical {
    background: $primary; border: 2px solid $surface; width: 14px; height: 14px;
    margin: 0 -7px; border-radius: 9px;
}
QSlider::handle:disabled { background: $border_strong; }
QSlider::sub-page:disabled { background: $border_strong; }

/* ---- progress -------------------------------------------------------- */
QProgressBar {
    background: $surface_alt; border: 1px solid $border; border-radius: 8px;
    min-height: 16px; max-height: 16px; text-align: center; font-size: 11px; color: $text;
}
QProgressBar::chunk { background: $progress_fill; border-radius: 7px; }

/* ---- group boxes ----------------------------------------------------- */
QGroupBox {
    background: $surface; border: 1px solid $border; border-radius: 10px;
    margin-top: 12px; padding: 16px 12px 12px 12px; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 6px; color: $text_muted; background: $background;
}

/* ---- tool tips, menus, bars ------------------------------------------ */
QToolTip {
    background: $surface; color: $text; border: 1px solid $border_strong;
    padding: 4px 8px; border-radius: 4px;
}
QMenu {
    background: $surface; border: 1px solid $border; border-radius: 8px; padding: 4px;
}
QMenu::item { padding: 6px 24px 6px 12px; border-radius: 4px; background: transparent; }
QMenu::item:selected { background: $selection; color: $text; }
QMenu::item:disabled { color: $text_disabled; background: transparent; }
QMenu::icon { padding-left: 8px; }
QMenu::separator { height: 1px; background: $border; margin: 4px 8px; }
QMenu::indicator { width: 14px; height: 14px; margin-left: 6px; }
QMenuBar { background: $surface; border-bottom: 1px solid $border; padding: 2px 4px; }
QMenuBar::item { padding: 5px 10px; background: transparent; border-radius: 4px; }
QMenuBar::item:selected { background: $surface_alt; }
QMenuBar::item:pressed { background: $selection; }
QStatusBar { background: $surface; border-top: 1px solid $border; color: $text_muted; }
QStatusBar::item { border: none; }
QStatusBar QLabel { color: $text_muted; }
QToolBar { background: $surface; border: none; border-bottom: 1px solid $border; spacing: 4px; padding: 4px; }
QToolBar::separator { background: $border; width: 1px; margin: 4px 6px; }
QSplitter::handle { background: $border; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
QSplitter::handle:hover { background: $primary; }
QDockWidget { color: $text; }
QDockWidget::title { background: $surface_alt; padding: 6px 8px; }
""")


def stylesheet(palette: Palette, asset_dir: Optional[Path] = None) -> str:
    """The complete application style sheet for ``palette``."""
    folder = asset_dir if asset_dir is not None else ensure_assets(palette)
    values = {name: getattr(palette, name) for name in _COLOR_FIELDS}
    values.update(
        asset_dir=Path(folder).as_posix(),
        primary_soft=palette.tint("primary", 0.12),
        danger_soft=palette.tint("danger", 0.12),
        row_hover=palette.tint("primary", 0.06),
        progress_fill=palette.tint("primary", 0.40),
    )
    return _QSS.substitute(values)


def qpalette(palette: Palette) -> QPalette:
    """A ``QPalette`` for ``palette`` (so non-styled widgets match too)."""
    qp = QPalette()
    roles = {
        QPalette.ColorRole.Window: palette.background,
        QPalette.ColorRole.WindowText: palette.text,
        QPalette.ColorRole.Base: palette.surface,
        QPalette.ColorRole.AlternateBase: palette.surface_alt,
        QPalette.ColorRole.Text: palette.text,
        QPalette.ColorRole.Button: palette.surface,
        QPalette.ColorRole.ButtonText: palette.text,
        QPalette.ColorRole.BrightText: palette.primary_text,
        QPalette.ColorRole.Highlight: palette.selection,
        QPalette.ColorRole.HighlightedText: palette.text,
        QPalette.ColorRole.ToolTipBase: palette.surface,
        QPalette.ColorRole.ToolTipText: palette.text,
        QPalette.ColorRole.PlaceholderText: palette.text_muted,
        QPalette.ColorRole.Link: palette.primary,
        QPalette.ColorRole.LinkVisited: palette.primary_hover,
        QPalette.ColorRole.Light: palette.surface,
        QPalette.ColorRole.Midlight: palette.surface_alt,
        QPalette.ColorRole.Mid: palette.border,
        QPalette.ColorRole.Dark: palette.border_strong,
        QPalette.ColorRole.Shadow: "#000000",
        QPalette.ColorRole.Accent: palette.primary,
    }
    for role, color in roles.items():
        qp.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText,
                 QPalette.ColorRole.ButtonText, QPalette.ColorRole.HighlightedText):
        qp.setColor(QPalette.ColorGroup.Disabled, role, QColor(palette.text_disabled))
    return qp


# ----------------------------------------------------------------------
# Applying it
# ----------------------------------------------------------------------
_current: Palette = LIGHT


def current_palette() -> Palette:
    """The palette in force (light until ``apply_theme`` says otherwise)."""
    return _current


class ThemeController(QObject):
    """Owns the theme mode of an application and keeps it in sync with the OS.

    One controller exists per ``QApplication`` (see ``get_controller``). It is
    a child of the application, so its connection to the OS colour-scheme
    signal goes away with it.
    """

    palette_changed = pyqtSignal(object)    # the new ``Palette``

    def __init__(self, app: QApplication):
        super().__init__(app)
        self._app = app
        self._mode = "auto"
        self._fusion_set = False
        self._scheme = self._os_scheme()
        self._original = (
            QPalette(app.palette()), QFont(app.font()), app.styleSheet(),
            app.style().objectName(),
        )
        hints = QGuiApplication.styleHints()
        signal = getattr(hints, "colorSchemeChanged", None)
        if signal is not None:     # Qt 6.5+
            signal.connect(self._on_scheme_changed)

    @staticmethod
    def _os_scheme() -> Qt.ColorScheme:
        hints = QGuiApplication.styleHints()
        getter = getattr(hints, "colorScheme", None)
        return getter() if getter is not None else Qt.ColorScheme.Unknown

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def palette(self) -> Palette:
        return palette_for(self._mode, self._scheme)

    def set_mode(self, mode: str) -> Palette:
        """Choose ``"auto"``, ``"light"`` or ``"dark"`` and apply it."""
        if mode not in MODES:
            raise ValueError(f"Unknown theme mode: {mode!r}")
        self._mode = mode
        return self.apply()

    def _on_scheme_changed(self, scheme) -> None:
        """The OS switched between light and dark."""
        self._scheme = scheme
        if self._mode == "auto":
            self.apply()

    def apply(self) -> Palette:
        global _current
        palette = self.palette
        _current = palette
        app = self._app
        if not self._fusion_set:   # (style() is a style-sheet proxy once styled)
            app.setStyle("Fusion")
            self._fusion_set = True
        font = QFont(app.font())
        font.setFamily(font_family())
        font.setPixelSize(TypeScale.BODY)
        app.setFont(font)
        app.setPalette(qpalette(palette))
        app.setStyleSheet(stylesheet(palette))
        self.palette_changed.emit(palette)
        return palette

    def reset(self) -> None:
        """Undo the theme: platform style, palette, font and style sheet."""
        global _current
        original_palette, original_font, original_sheet, original_style = self._original
        app = self._app
        app.setStyleSheet(original_sheet)
        if self._fusion_set and original_style:
            app.setStyle(original_style)
            self._fusion_set = False
        app.setPalette(original_palette)
        app.setFont(original_font)
        _current = LIGHT


def get_controller(app: Optional[QApplication] = None) -> Optional[ThemeController]:
    """The theme controller of the application, if a theme was applied."""
    app = app or QApplication.instance()
    if app is None:
        return None
    return app.findChild(ThemeController)


def apply_theme(app: QApplication, mode: str = "auto") -> Palette:
    """Apply the workspace theme application-wide and return its palette.

    ``mode``: ``"auto"`` follows the OS colour scheme (and keeps following it
    live), ``"light"`` / ``"dark"`` pin it. Calling it again with another mode
    switches the theme.
    """
    controller = get_controller(app) or ThemeController(app)
    return controller.set_mode(mode)


def reset_theme(app: Optional[QApplication] = None) -> None:
    """Remove the workspace theme again (tests; the classic window)."""
    controller = get_controller(app)
    if controller is not None:
        controller.reset()
        controller.setParent(None)
        controller.deleteLater()


# ----------------------------------------------------------------------
# Small helpers for widgets
# ----------------------------------------------------------------------
def _repolish(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_class(widget: QWidget, name: str) -> None:
    """Set the ``class`` property styled by the style sheet (buttons)."""
    if widget.property("class") != name:
        widget.setProperty("class", name)
        _repolish(widget)


def set_role(widget: QWidget, role: str) -> None:
    """Set the ``role`` property (``display``, ``title``, ``subtitle``,
    ``body``, ``caption``, ``muted``) styled by the style sheet (labels)."""
    if widget.property("role") != role:
        widget.setProperty("role", role)
        _repolish(widget)


ROW_HEIGHT = 32


def style_table_view(view: QTableView, row_height: int = ROW_HEIGHT) -> None:
    """The workspace table look: comfortable rows, zebra, no grid clutter."""
    view.setAlternatingRowColors(True)
    view.setShowGrid(False)
    view.setWordWrap(False)
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    view.verticalHeader().setVisible(False)
    view.verticalHeader().setDefaultSectionSize(row_height)
    view.verticalHeader().setMinimumSectionSize(row_height)
    view.horizontalHeader().setHighlightSections(False)
    view.horizontalHeader().setDefaultAlignment(
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    )
    view.horizontalHeader().setMinimumHeight(row_height + 4)
    view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
