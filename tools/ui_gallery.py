"""
tools/ui_gallery.py
Visual QA for the workspace design system: renders every ``ui_kit`` widget in
representative states and saves one PNG per group.

    python tools/ui_gallery.py --out DIR [--dark]

Nothing appears on screen: every page is a top-level widget with
``WA_DontShowOnScreen``, which Qt still lays out and paints, so ``grab()``
gives the real rendering. Run it with the platform's default Qt plugin (the
``offscreen`` plugin has no fonts, so text would be invisible).

``render_gallery`` is importable (the tests call it under ``offscreen``).
"""
import argparse
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2                                            # noqa: E402
import numpy as np                                    # noqa: E402
from PyQt6.QtCore import QCoreApplication, QSize, Qt  # noqa: E402
from PyQt6.QtGui import QStandardItem, QStandardItemModel  # noqa: E402
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,  # noqa: E402
                             QFormLayout, QGridLayout, QHBoxLayout, QLabel,
                             QLineEdit, QProgressBar, QRadioButton, QSlider,
                             QSpinBox, QTableView, QTabWidget, QToolButton,
                             QVBoxLayout, QWidget)

from gui.workspace import theme                       # noqa: E402
from gui.workspace.thumbnails import ThumbnailLoader  # noqa: E402
from gui.workspace.ui_kit import (Callout, Card, CollapsibleSection,  # noqa: E402
                                  DangerButton, DropZone, EmptyState, Filmstrip,
                                  FilmstripItem, FooterBar, GhostButton, InlineHelp,
                                  KeyHint, Label, ParadigmPicker, PrimaryButton,
                                  SecondaryButton, StageHeader, StatusChip, StepInfo,
                                  Stepper)

PAGE_WIDTH = 1100
MARGIN = 24

WORKFLOW = [
    ("videos", "Videos", "video"),
    ("arena", "Arena & zones", "zones"),
    ("detection", "Detection", "target"),
    ("trials", "Trials", "clock"),
    ("review", "Review & run", "list"),
    ("results", "Results", "chart"),
]


def make_steps(statuses: List[str], reasons: Optional[List[str]] = None) -> List[StepInfo]:
    reasons = reasons or [""] * len(WORKFLOW)
    return [StepInfo(i, label, icon, status, reason)
            for (i, label, icon), status, reason in zip(WORKFLOW, statuses, reasons)]


def _page(width: int = PAGE_WIDTH) -> "tuple[QWidget, QVBoxLayout]":
    page = QWidget()
    page.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    page.setAutoFillBackground(True)
    page.setFixedWidth(width)
    layout = QVBoxLayout(page)
    layout.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
    layout.setSpacing(theme.Spacing.MD)
    return page, layout


def _caption(text: str) -> QLabel:
    return Label(text, "caption")


def _row(*widgets: QWidget, stretch: bool = True) -> QWidget:
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(theme.Spacing.SM)
    for widget in widgets:
        row.addWidget(widget)
    if stretch:
        row.addStretch(1)
    return holder


def write_demo_videos(folder: Path, count: int = 5) -> List[Path]:
    """Tiny coloured clips so the filmstrip has real first frames."""
    paths = []
    for index in range(count):
        path = folder / f"mouse_{index + 1:02d}_trial.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 180))
        base = np.zeros((180, 320, 3), np.uint8)
        hue = int(index * 180 / count)
        gradient = np.linspace(40, 200, 320, dtype=np.uint8)
        base[:] = cv2.cvtColor(
            np.dstack([np.full((180, 320), hue, np.uint8),
                       np.full((180, 320), 110, np.uint8),
                       np.tile(gradient, (180, 1))]), cv2.COLOR_HSV2BGR)
        cv2.rectangle(base, (60, 30), (260, 150), (235, 235, 235), 2)
        cv2.circle(base, (150 + index * 12, 95), 9, (20, 20, 20), -1)
        for _ in range(3):
            writer.write(base)
        writer.release()
        paths.append(path)
    return paths


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------
def page_stepper(ctx) -> QWidget:
    page, layout = _page()
    layout.addWidget(_caption("Stepper at 1100 px: done, current, attention, todo, disabled"))
    layout.addWidget(Stepper(make_steps(
        ["done", "current", "attention", "todo", "disabled", "disabled"],
        ["3 videos added", "Draw the arena and zones", "2 videos have no detection settings",
         "", "Finish the earlier steps first", "Run the analysis to see results"])))
    layout.addWidget(_caption("All done, last step current"))
    layout.addWidget(Stepper(make_steps(["done"] * 5 + ["current"])))
    layout.addWidget(_caption("First step current, the rest not available yet"))
    layout.addWidget(Stepper(make_steps(["current"] + ["disabled"] * 5)))
    for width, text in ((820, "At 820 px the captions go"),
                        (620, "At 620 px labels are elided"),
                        (380, "At 380 px only the badges remain")):
        layout.addWidget(_caption(text))
        stepper = Stepper(make_steps(["done", "current", "todo", "todo", "todo", "todo"]))
        stepper.setFixedWidth(width)
        layout.addWidget(stepper)
    focused = Stepper(make_steps(["done", "current", "todo", "todo", "todo", "todo"]))
    focused.setFocus()
    layout.addWidget(_caption("Hovered step"))
    hovered = Stepper(make_steps(["done", "done", "current", "todo", "todo", "todo"]))
    hovered._hover = 4
    layout.addWidget(hovered)
    return page


def page_chrome(ctx) -> QWidget:
    page, layout = _page()
    layout.addWidget(StageHeader(
        "Add your videos", "Drop the recordings you want to analyse; you can add more later.",
        "video", show_help=True))
    header = StageHeader(
        "Arena & zones", "Mark the arena once, then copy it to the other videos.", "zones",
        show_help=True)
    header.add_action(SecondaryButton("Copy to all videos", "plus"))
    header.add_action(PrimaryButton("Draw zones", "zones"))
    layout.addWidget(header)
    layout.addSpacing(theme.Spacing.SM)
    layout.addWidget(_caption("Footer: enabled, status text and chip"))
    footer = FooterBar()
    footer.set_next("Arena & zones")
    footer.set_status("3 videos ready", ("ok", "Ready"))
    layout.addWidget(footer)
    layout.addWidget(_caption("Footer: Next disabled with a reason (tooltip), warning chip"))
    footer = FooterBar()
    footer.set_next("Detection", enabled=False, reason="Draw the arena on at least one video")
    footer.set_status("No arena drawn yet", ("warning", "Needs attention"))
    layout.addWidget(footer)
    layout.addWidget(_caption("Footer: last stage, running"))
    footer = FooterBar()
    footer.add_extra(SecondaryButton("Save project", "check"))
    footer.set_next("Run analysis", prefix="", arrow=False)
    footer.set_status("Analysing video 2 of 6", ("running", "Running"))
    layout.addWidget(footer)
    return page


def page_buttons(ctx) -> QWidget:
    page, layout = _page()
    layout.addWidget(_caption("Buttons: normal, with icon, disabled"))
    for factory, text, icon_name in ((PrimaryButton, "Primary", "play"),
                                     (SecondaryButton, "Secondary", "folder"),
                                     (GhostButton, "Ghost", "search"),
                                     (DangerButton, "Remove", "trash")):
        off = factory(text + " (off)")
        off.setEnabled(False)
        layout.addWidget(_row(factory(text), factory(text, icon_name), off))
    layout.addWidget(_caption("Status chips"))
    layout.addWidget(_row(
        StatusChip("ok", "Done"), StatusChip("warning", "Needs review"),
        StatusChip("error", "Failed"), StatusChip("info", "Calibrated"),
        StatusChip("neutral", "New"), StatusChip("running", "Analysing 42%")))
    layout.addWidget(_caption("Key hints"))
    layout.addWidget(_row(KeyHint("S"), KeyHint("Space"), KeyHint("Ctrl+Z"), Label("to save, play, undo")))
    layout.addWidget(_caption("Callouts"))
    layout.addWidget(Callout("tip", "<b>Tip:</b> press <b>S</b> to start a new scoring segment."))
    layout.addWidget(Callout("info", "Zones are saved with the project and shared by every video."))
    layout.addWidget(Callout("warning", "Two videos have a different frame size from the arena template.",
                             title="Check your videos"))
    layout.addWidget(Callout("error", "The video cannot be opened. It may have been moved or deleted."))
    layout.addWidget(Callout("success", "All videos were analysed without quality warnings."))
    layout.addWidget(InlineHelp("Detection works best when the mouse is darker than the floor.",
                                dismissible=True))
    return page


def page_containers(ctx) -> QWidget:
    page, layout = _page()
    card = Card("Detection settings", "Applies to every selected video.")
    card.add_header_widget(GhostButton("Reset", "reset"))
    card.add_widget(Label("Threshold and minimum size decide what counts as the mouse.", "body", wrap=True))
    section = CollapsibleSection("Advanced options", expanded=False, summary="3 changed", animated=False)
    section.body.addWidget(Label("Background frames, smoothing and sub-pixel tracking.", "muted", wrap=True))
    card.add_widget(section)
    expanded = CollapsibleSection("Calibration", expanded=True, animated=False)
    expanded.body.addWidget(Label("Arena width: 40 cm", "body"))
    expanded.body.addWidget(Label("Arena height: 40 cm", "body"))
    card.add_widget(expanded)
    layout.addWidget(card)
    empty = EmptyState("video", "No videos yet",
                       "Add the recordings you want to analyse. Everything stays on this computer.")
    empty.add_action("Add videos", icon_name="plus")
    empty.add_action("Open a project", icon_name="folder")
    holder = Card()
    holder.add_widget(empty)
    layout.addWidget(holder)
    return page


def page_dropzone(ctx) -> QWidget:
    page, layout = _page()
    layout.addWidget(_caption("Idle"))
    layout.addWidget(DropZone())
    layout.addWidget(_caption("A droppable item is held over the zone"))
    dragging = DropZone()
    dragging._set_dragging(True)
    layout.addWidget(dragging)
    layout.addWidget(_caption("Compact, when videos are already listed"))
    compact = DropZone()
    compact.set_compact(True)
    layout.addWidget(compact)
    return page


def page_paradigms(ctx) -> QWidget:
    page, layout = _page()
    layout.addWidget(_caption("Paradigm picker at 1100 px, one chosen"))
    picker = ParadigmPicker()
    picker.set_selected("Elevated Plus Maze")
    layout.addWidget(picker)
    layout.addWidget(_caption("Narrower (600 px)"))
    narrow = ParadigmPicker()
    narrow.setFixedWidth(600)
    narrow.set_selected("Y-Maze")
    layout.addWidget(narrow)
    return page


def page_filmstrip(ctx) -> QWidget:
    page, layout = _page()
    badges = [("Done", "ok"), ("Needs review", "warning"), ("Failed", "error"),
              ("Configured", "info"), ("New", "neutral")]
    items = [FilmstripItem(f"k{i}", path, badge_text=badges[i][0], badge_kind=badges[i][1])
             for i, path in enumerate(ctx["videos"])]
    items.append(FilmstripItem("missing", Path("no_such_video.mp4"), badge_text="Missing", badge_kind="error"))
    layout.addWidget(_caption("Horizontal strip"))
    strip = Filmstrip(Qt.Orientation.Horizontal, ctx["loader"])
    strip.set_items(items)
    strip.select("k1")
    layout.addWidget(strip)
    layout.addWidget(_caption("Compact list"))
    compact = Filmstrip(Qt.Orientation.Horizontal, ctx["loader"])
    compact.set_items(items)
    compact.set_compact(True)
    compact.select("k2")
    layout.addWidget(compact)
    layout.addWidget(_caption("Vertical, thumbnails and compact"))
    both = QWidget()
    columns = QHBoxLayout(both)
    columns.setContentsMargins(0, 0, 0, 0)
    for compact_flag in (False, True):
        vertical = Filmstrip(Qt.Orientation.Vertical, ctx["loader"], title="Videos")
        vertical.set_items(items)
        vertical.set_compact(compact_flag)
        vertical.select("k0")
        vertical.setFixedHeight(380)
        columns.addWidget(vertical)
    layout.addWidget(both)
    return page


def page_controls(ctx) -> QWidget:
    page, layout = _page()
    grid = QGridLayout()
    grid.setHorizontalSpacing(theme.Spacing.LG)
    form = QFormLayout()
    form.addRow("Name", QLineEdit("Open field 1"))
    invalid = QLineEdit()
    invalid.setPlaceholderText("Required")
    invalid.setProperty("invalid", True)
    form.addRow("Invalid", invalid)
    combo = QComboBox()
    combo.addItems(["Y-Maze", "Elevated Plus Maze"])
    form.addRow("Paradigm", combo)
    spin = QSpinBox()
    spin.setValue(12)
    form.addRow("Threshold", spin)
    double = QDoubleSpinBox()
    double.setValue(2.5)
    form.addRow("Scale (cm)", double)
    disabled = QLineEdit("Disabled")
    disabled.setEnabled(False)
    form.addRow("Locked", disabled)
    checked = QCheckBox("Remember this choice")
    checked.setChecked(True)
    form.addRow(checked)
    form.addRow(QCheckBox("Search sub-folders"))
    on = QRadioButton("Per video")
    on.setChecked(True)
    form.addRow(on)
    form.addRow(QRadioButton("Same for all"))
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setValue(45)
    form.addRow("Speed", slider)
    progress = QProgressBar()
    progress.setValue(62)
    form.addRow(progress)
    left = QWidget()
    left.setLayout(form)
    grid.addWidget(left, 0, 0)
    table = QTableView()
    model = QStandardItemModel(6, 4)
    model.setHorizontalHeaderLabels(["Status", "Video", "Settings", "Notes"])
    for row in range(6):
        for column, text in enumerate(["Done", f"mouse_{row + 1:02d}.mp4", "Template", ""]):
            model.setItem(row, column, QStandardItem(text))
    table.setModel(model)
    theme.style_table_view(table)
    table.selectRow(2)
    table.setMinimumSize(460, 280)
    grid.addWidget(table, 0, 1)
    layout.addLayout(grid)
    tabs = QTabWidget()
    tabs.addTab(Label("Tab content", "body"), "Trials")
    tabs.addTab(QWidget(), "Results")
    layout.addWidget(tabs)
    return page


def page_icons(ctx) -> QWidget:
    page, layout = _page()
    grid = QGridLayout()
    grid.setSpacing(theme.Spacing.SM)
    for index, name in enumerate(theme.ICON_NAMES):
        button = QToolButton()
        button.setIcon(theme.icon(name))
        button.setIconSize(QSize(24, 24))
        cell = QWidget()
        column = QVBoxLayout(cell)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        column.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)
        label = Label(name, "caption")
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        column.addWidget(label)
        grid.addWidget(cell, index // 10, index % 10)
    layout.addLayout(grid)
    return page


def page_stage_mockup(ctx) -> QWidget:
    """A whole stage assembled from the kit: what stage authors will build."""
    page = QWidget()
    page.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    page.setAutoFillBackground(True)
    page.setFixedSize(PAGE_WIDTH, 760)
    outer = QVBoxLayout(page)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    top = QWidget()
    top_layout = QVBoxLayout(top)
    top_layout.setContentsMargins(MARGIN, theme.Spacing.MD, MARGIN, theme.Spacing.SM)
    top_layout.addWidget(Stepper(make_steps(["done", "current", "todo", "todo", "disabled", "disabled"],
                                            ["3 videos added", "", "", "", "", ""])))
    outer.addWidget(top)
    body = QWidget()
    body_layout = QVBoxLayout(body)
    body_layout.setContentsMargins(MARGIN, theme.Spacing.MD, MARGIN, theme.Spacing.MD)
    body_layout.setSpacing(theme.Spacing.MD)
    body_layout.addWidget(StageHeader(
        "Choose the paradigm", "Pick the test these videos belong to. This decides which zones and "
        "results you get.", "settings", show_help=True))
    picker = ParadigmPicker()
    picker.set_selected("Freestyle / Open Field")
    body_layout.addWidget(picker)
    body_layout.addWidget(Callout("tip", "You can change the paradigm later; your zones are kept when "
                                         "the new paradigm uses the same ones."))
    section = CollapsibleSection("More options", expanded=False, animated=False)
    section.body.addWidget(Label("Recursive folder search, arena size and frame rate.", "muted"))
    body_layout.addWidget(section)
    body_layout.addStretch(1)
    outer.addWidget(body, 1)
    footer = FooterBar()
    footer.set_next("Detection")
    footer.set_status("Open field selected", ("ok", "Ready"))
    outer.addWidget(footer)
    return page


PAGES: Dict[str, Callable] = {
    "stepper": page_stepper,
    "stage_chrome": page_chrome,
    "buttons_chips_callouts": page_buttons,
    "containers": page_containers,
    "dropzone": page_dropzone,
    "paradigms": page_paradigms,
    "filmstrip": page_filmstrip,
    "controls": page_controls,
    "icons": page_icons,
    "stage_mockup": page_stage_mockup,
}


def render_gallery(out_dir, dark: bool = False,
                   app: Optional[QApplication] = None) -> List[Path]:
    """Render every page to ``out_dir`` and return the PNG paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    app = app or QApplication.instance() or QApplication(sys.argv[:1])
    theme.apply_theme(app, "dark" if dark else "light")
    suffix = "_dark" if dark else ""
    written: List[Path] = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        videos = write_demo_videos(Path(tmp))
        loader = ThumbnailLoader()
        ctx = {"videos": videos, "loader": loader}
        try:
            for name, factory in PAGES.items():
                page = factory(ctx)
                page.show()
                QCoreApplication.processEvents()
                loader.wait_idle()
                QCoreApplication.processEvents()
                target = out / f"{name}{suffix}.png"
                if not page.grab().save(str(target), "PNG"):
                    raise OSError(f"Could not write {target}")
                written.append(target)
                page.close()
                page.deleteLater()
        finally:
            loader.shutdown()
            loader.deleteLater()
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render the workspace UI kit to PNG files.")
    parser.add_argument("--out", required=True, help="folder for the PNGs")
    parser.add_argument("--dark", action="store_true", help="render the dark theme")
    options = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        paths = render_gallery(options.out, dark=options.dark, app=app)
    finally:
        theme.reset_theme(app)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
