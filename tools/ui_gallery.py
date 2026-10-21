"""
tools/ui_gallery.py
Visual QA for the workspace design system: renders every ``ui_kit`` widget in
representative states and saves one PNG per group.

    python tools/ui_gallery.py --out DIR [--dark]
    python tools/ui_gallery.py --workspace --out DIR [--dark] [--size 1366x768 ...]

The first form renders the design-system widgets; ``--workspace`` renders the
whole workspace window instead: every stage (the Videos stage in each of its
states, the details drawer, the stages hosting the old tabs) at each size,
as ``ws_<page>_<W>x<H>[_dark].png``.

Nothing appears on screen: every page is a top-level widget with
``WA_DontShowOnScreen``, which Qt still lays out and paints, so ``grab()``
gives the real rendering. Run it with the platform's default Qt plugin (the
``offscreen`` plugin has no fonts, so text would be invisible).

``render_gallery`` is importable (the tests call it under ``offscreen``).
"""
import argparse
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


# ----------------------------------------------------------------------
# The whole workspace (--workspace)
# ----------------------------------------------------------------------
WORKSPACE_SIZES = ((1366, 768), (1920, 1080))


def _write_clip(path: Path, frames: int = 150, fps: float = 10.0, seed: int = 0,
                scale: float = 1.0) -> Path:
    """A short clip of a mouse-like dot walking in a box (real first frame).

    ``scale`` makes the same scene bigger or smaller (another camera
    resolution); everything is drawn 320x180 and scaled.
    """
    size = (round(320 * scale), round(180 * scale))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for index in range(frames):
        frame = np.full((180, 320, 3), 205, np.uint8)
        cv2.rectangle(frame, (30, 15), (290, 165), (150, 150, 150), 2)
        angle = (index + seed * 40) / 14.0
        x = int(160 + 90 * np.cos(angle))
        y = int(90 + 50 * np.sin(angle * 1.3))
        cv2.circle(frame, (x, y), 8, (35, 35, 35), -1)
        if scale != 1.0:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        writer.write(frame)
    writer.release()
    return path


def _demo_settings(paradigm: str = "Y-Maze"):
    """Valid settings for the demo clips: three arms and a centre."""
    from core.models import AppConfig, BatchSettings
    from mazes import create_maze
    from roi_manager import ROIManager
    manager = ROIManager(source_dimensions=(320, 180))
    boxes = {
        "arm_a": [[40, 25], [140, 25], [140, 70], [40, 70]],
        "arm_b": [[180, 25], [280, 25], [280, 70], [180, 70]],
        "arm_c": [[100, 110], [220, 110], [220, 160], [100, 160]],
        "center": [[130, 75], [190, 75], [190, 105], [130, 105]],
    }
    for name, _label in create_maze(paradigm).get_roi_definitions():
        manager.add_roi(name, np.array(boxes.get(name, boxes["center"])))
    return BatchSettings(
        manager, AppConfig().detection, 0.0, paradigm_id=paradigm,
        algorithm_parameters=create_maze(paradigm).get_algorithm_parameters(),
    )


def _quiet_prompts():
    """A ``Prompts`` that never opens a dialog (the gallery must not block)."""
    from gui.workspace.prompts import DISCARD, Prompts

    class QuietPrompts(Prompts):
        def info(self, title, text):
            pass

        def warn(self, title, text):
            pass

        def error(self, title, text):
            pass

        def about(self, title, text):
            pass

        def confirm(self, *args, **kwargs):
            return True

        def ask_save_changes(self, project_name):
            return DISCARD

        def ask_recovery(self, project_name, autosave):
            return "saved"

        def ask_text(self, *args, **kwargs):
            return None

        def ask_item(self, *args, **kwargs):
            return None

        def open_file(self, *args, **kwargs):
            return None

        def open_files(self, *args, **kwargs):
            return []

        def save_file(self, *args, **kwargs):
            return None

        def choose_folder(self, *args, **kwargs):
            return None

        def open_folder(self, path):
            return True

        def reveal(self, path):
            return True

        def open_url(self, url):
            return True

        def copy_text(self, text):
            pass

    return QuietPrompts(None)


class WorkspaceDemo:
    """A throw-away folder with clips, projects and settings, and the window."""

    def __init__(self, root: Path, size: Tuple[int, int]):
        from core.project import Project
        from core.project_io import save_project
        from gui.app_settings import AppSettings
        from gui.workspace.window import WorkspaceWindow
        self.root = root
        self.size = size
        folder = root / "videos"
        folder.mkdir(exist_ok=True)
        names = ["mouse_01_trial1.avi", "mouse_02_trial1.avi", "mouse_03_trial1.avi",
                 "mouse_04_trial1.avi", "mouse_05_trial1.avi", "mouse_06_trial1.avi"]
        self.clips = [_write_clip(folder / n, frames=120 + 30 * i, seed=i)
                      for i, n in enumerate(names)]
        self.settings = AppSettings.for_file(root / "settings.ini")
        # A study with videos, a template, trial times and one missing file.
        study = Project(name="Y-maze cohort 1", paradigm="Y-Maze")
        study.set_template(_demo_settings())
        study.add_videos(self.clips)
        study.add_metadata_column("Group")
        for index, entry in enumerate(study.videos):
            study.set_metadata_value(
                entry.key, "Group", "Control" if index % 2 == 0 else "Treated")
        study.set_trial_protocol(300.0, "per_video")
        for index, entry in enumerate(study.videos[:4]):
            study.set_trial(entry.key, start=4.0 + 2 * index)
        study.set_verified([study.videos[0].key, study.videos[1].key], True)
        study.videos[0].last_status = "done"
        study.videos[2].last_status = "warning"
        self.study_path = save_project(study, root / "Y-maze cohort 1.bmaproj")
        self.clips[5].unlink()
        # Two more projects so the start page has recent projects to show.
        for name in ("Open field pilot", "Tail suspension March"):
            other = Project(name=name, paradigm="Freestyle / Open Field")
            self.settings.add_recent_project(save_project(other, root / f"{name}.bmaproj"))
        self.settings.add_recent_project(self.study_path)
        self.settings.clear_last_project()
        template = root / "my_old_template.json"
        _demo_settings().save_to_file(template)
        self.settings.add_recent_file(template)
        self.window = WorkspaceWindow(self.settings, _quiet_prompts())
        self.window.drawer.animation_ms = 0
        self.window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        self.window.resize(*size)
        self.window.show()
        self.settle()

    def settle(self) -> None:
        for _ in range(3):
            QCoreApplication.processEvents()
            self.window.thumbnails.wait_idle()
        QCoreApplication.processEvents()
        self.window.refresh_navigation()

    def grab(self, target: Path) -> Path:
        self.settle()
        if not self.window.grab().save(str(target), "PNG"):
            raise OSError(f"Could not write {target}")
        return target

    def close(self) -> None:
        self.window._skip_close_prompt = True
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.processEvents()


def ws_welcome(demo: WorkspaceDemo) -> None:
    demo.window.start()


def ws_paradigm(demo: WorkspaceDemo) -> None:
    from core.project import Project
    from core.project_io import save_project
    path = save_project(Project(name="New study"), demo.root / "New study.bmaproj")
    demo.window.open_path(path)


def ws_videos_empty(demo: WorkspaceDemo) -> None:
    ws_paradigm(demo)
    demo.window.change_paradigm("Y-Maze")


def ws_videos(demo: WorkspaceDemo) -> None:
    demo.window.open_path(demo.study_path)
    window = demo.window
    window.go_to_stage("videos", force=True)
    keys = window.project.keys()
    window.video_panel.select_keys(keys[1:2])
    window.video_panel.flush_selection()


def ws_videos_drawer(demo: WorkspaceDemo) -> None:
    ws_videos(demo)
    keys = demo.window.project.keys()
    demo.window.show_details(keys[2:3])


def ws_videos_hover(demo: WorkspaceDemo) -> None:
    """The buttons that float over the row under the mouse."""
    ws_videos(demo)
    demo.settle()
    view = demo.window.video_panel.view
    view.show_row_actions(view.model().index(5, 0))      # the missing video


def ws_videos_change(demo: WorkspaceDemo) -> None:
    ws_videos(demo)
    demo.window.videos_stage.change_paradigm()


def _stage_page(stage_id: str):
    def render(demo: WorkspaceDemo) -> None:
        ws_videos(demo)
        demo.window.go_to_stage(stage_id, force=True)
    return render


def ws_trials_protocol(demo: WorkspaceDemo) -> None:
    """The trials page with the protocol card open and a cell being refused."""
    ws_videos(demo)
    window = demo.window
    window.go_to_stage("trials", force=True)
    stage = window.stage("trials")
    stage.protocol.section.set_expanded(True)
    keys = window.project.keys()
    stage.select_row(keys[4])
    # An end before the start (kept, red) and text that is not a time.
    stage.model.setData(stage.model.index(stage.model.row_of(keys[3]), 3), "0:05")
    stage.model.setData(stage.model.index(stage.model.row_of(keys[4]), 2), "abc")
    stage.player.wait_idle()


def _tst_demo_project(demo: WorkspaceDemo):
    """Replace the demo study by a tail suspension study of three clips."""
    from core.models import AppConfig, BatchSettings
    from core.project import Project
    from core.project_io import save_project
    from mazes import create_maze
    from roi_manager import ROIManager
    maze = create_maze("Tail Suspension Test")
    maze.configure_mice(3)
    manager = ROIManager(source_dimensions=(320, 180))
    for index, (name, _label) in enumerate(maze.get_roi_definitions()):
        x = 20 + index * 100
        manager.add_roi(name, np.array([[x, 20], [x + 80, 20], [x + 80, 160], [x, 160]]))
    study = Project(name="Tail suspension cohort", paradigm="Tail Suspension Test")
    study.set_template(BatchSettings(
        manager, AppConfig().detection, 0.0, paradigm_id="Tail Suspension Test",
        algorithm_parameters=maze.get_algorithm_parameters(),
    ))
    study.add_videos(demo.clips[:3])
    first, second, _third = (v.key for v in study.videos)
    study.set_trial(first, per_roi_times={
        "mouse_1": (1.0, 7.0), "mouse_2": (2.0, 8.0), "mouse_3": (3.0, 9.0)})
    study.set_trial(second, per_roi_times={"mouse_1": (0.0, 6.0)})
    return save_project(study, demo.root / "Tail suspension cohort.bmaproj")


def ws_trials_tst(demo: WorkspaceDemo) -> None:
    """Tail suspension: one window per mouse, grouped by video."""
    window = demo.window
    window.open_path(_tst_demo_project(demo))
    window.go_to_stage("trials", force=True)
    stage = window.stage("trials")
    stage.select_row(window.project.keys()[0], "mouse_2")
    stage.player.wait_idle()


def ws_trials_empty(demo: WorkspaceDemo) -> None:
    """No videos yet: the empty state points back to the Videos page."""
    ws_videos_empty(demo)
    demo.window.go_to_stage("trials", force=True)


def _arena_stage(demo: WorkspaceDemo, extra_clips: Sequence[Path] = ()):
    """Open the study on the Arena & zones stage and let the video facts in."""
    ws_videos(demo)
    if extra_clips:
        demo.window.project.add_videos(extra_clips)
    demo.window.go_to_stage("arena", force=True)
    demo.settle()
    return demo.window.stage("arena")


def ws_arena_following(demo: WorkspaceDemo) -> None:
    """One video, template zones read-only over its own frame."""
    stage = _arena_stage(demo)
    stage.set_mode("video")
    stage.pick_video(demo.window.project.keys()[1])


def ws_arena_adjusted(demo: WorkspaceDemo) -> None:
    """A video with zones of its own, editable on its frame."""
    stage = _arena_stage(demo)
    stage.set_mode("video")
    stage.pick_video(demo.window.project.keys()[2])
    stage.adjust_video()


def ws_arena_check(demo: WorkspaceDemo) -> None:
    """A video with another resolution: "Check alignment"."""
    wide = _write_clip(demo.root / "videos" / "mouse_07_other_camera.avi", frames=90,
                       seed=6, scale=1.5)
    stage = _arena_stage(demo, [wide])
    stage.set_mode("video")
    stage.pick_video(demo.window.project.keys()[-1])


def ws_arena_options(demo: WorkspaceDemo) -> None:
    """The paradigm options and the calibration, opened."""
    stage = _arena_stage(demo)
    stage.params_section.set_expanded(True)
    demo.settle()
    stage.side.verticalScrollBar().setValue(stage.side.verticalScrollBar().maximum())


def ws_arena_openfield(demo: WorkspaceDemo) -> None:
    """The open field: its wider option panel, opened."""
    from core.project import Project
    from core.project_io import save_project
    study = Project(name="Open field cohort", paradigm="Freestyle / Open Field")
    study.set_template(_demo_settings("Freestyle / Open Field"))
    study.add_videos(demo.clips[:3])
    path = save_project(study, demo.root / "Open field cohort.bmaproj")
    demo.window.open_path(path)
    demo.window.go_to_stage("arena", force=True)
    demo.settle()
    stage = demo.window.stage("arena")
    stage.params_section.set_expanded(True)
    demo.settle()
    stage.side.verticalScrollBar().setValue(stage.side.verticalScrollBar().maximum())


def ws_arena_blank(demo: WorkspaceDemo) -> None:
    """A study whose template is not drawn yet."""
    ws_videos(demo)
    demo.window.project.set_template(None)
    demo.window.go_to_stage("arena", force=True)


WORKSPACE_PAGES: Dict[str, Callable[[WorkspaceDemo], None]] = {
    "welcome": ws_welcome,
    "paradigm": ws_paradigm,
    "videos_empty": ws_videos_empty,
    "videos": ws_videos,
    "videos_drawer": ws_videos_drawer,
    "videos_hover": ws_videos_hover,
    "videos_change": ws_videos_change,
    "arena": _stage_page("arena"),
    "arena_following": ws_arena_following,
    "arena_adjusted": ws_arena_adjusted,
    "arena_check": ws_arena_check,
    "arena_options": ws_arena_options,
    "arena_openfield": ws_arena_openfield,
    "arena_blank": ws_arena_blank,
    "detection": _stage_page("detection"),
    "trials": _stage_page("trials"),
    "trials_protocol": ws_trials_protocol,
    "trials_tst": ws_trials_tst,
    "trials_empty": ws_trials_empty,
    "review": _stage_page("review"),
    "results": _stage_page("results"),
}


def render_workspace(out_dir, dark: bool = False,
                     sizes: Sequence[Tuple[int, int]] = WORKSPACE_SIZES,
                     pages: Optional[Sequence[str]] = None,
                     app: Optional[QApplication] = None) -> List[Path]:
    """Render the workspace window, page by page and size by size.

    Each page is a fresh window in the demo state that page needs; the PNG is
    ``ws_<page>_<W>x<H>[_dark].png``. Returns the written paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    app = app or QApplication.instance() or QApplication(sys.argv[:1])
    theme.apply_theme(app, "dark" if dark else "light")
    suffix = "_dark" if dark else ""
    written: List[Path] = []
    for width, height in sizes:
        for name in (pages or WORKSPACE_PAGES):
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                demo = WorkspaceDemo(Path(tmp), (width, height))
                try:
                    WORKSPACE_PAGES[name](demo)
                    written.append(demo.grab(out / f"ws_{name}_{width}x{height}{suffix}.png"))
                finally:
                    demo.close()
    return written


def _parse_size(text: str) -> Tuple[int, int]:
    try:
        width, height = text.lower().split("x")
        return int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not WIDTHxHEIGHT") from exc


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render the workspace UI kit to PNG files.")
    parser.add_argument("--out", required=True, help="folder for the PNGs")
    parser.add_argument("--dark", action="store_true", help="render the dark theme")
    parser.add_argument("--workspace", action="store_true",
                        help="render the whole workspace window instead of the kit")
    parser.add_argument("--size", action="append", type=_parse_size, metavar="WxH",
                        help="window size for --workspace (repeatable; "
                             "default 1366x768 and 1920x1080)")
    parser.add_argument("--page", action="append", choices=sorted(WORKSPACE_PAGES),
                        help="only this --workspace page (repeatable)")
    options = parser.parse_args(argv)
    # PyQt aborts silently on an exception in a paint event or slot unless an
    # excepthook is installed; a QA run must say what went wrong instead.
    sys.excepthook = lambda *error: traceback.print_exception(*error)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        if options.workspace:
            paths = render_workspace(options.out, dark=options.dark,
                                     sizes=options.size or WORKSPACE_SIZES,
                                     pages=options.page, app=app)
        else:
            paths = render_gallery(options.out, dark=options.dark, app=app)
    finally:
        theme.reset_theme(app)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
