"""The keyboard / mouse cheat sheet shown by the editor's "?" overlay."""
from typing import List, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

# (heading, [(keys, what it does), ...]); also the source for the quickstart.
SHORTCUT_GROUPS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("Drawing", [
        ("Click", "Add a point to the new polygon"),
        ("Enter / double-click / right-click", "Close the polygon"),
        ("Esc", "Cancel the polygon in progress"),
        ("Backspace", "Remove the last point"),
        ("Alt (held)", "Turn point snapping off"),
    ]),
    ("Editing", [
        ("Drag a vertex", "Move it (snaps to neighbouring vertices)"),
        ("Drag a polygon", "Move the whole polygon"),
        ("Double-click an edge", "Insert a vertex"),
        ("Delete / right-click", "Delete the vertex or polygon"),
        ("Arrow keys", "Nudge the selection 1 px (Shift: 10 px)"),
        ("Blue / orange handle", "Rotate / scale the selected polygon"),
    ]),
    ("Reference line", [
        ("Click twice or drag", "Draw the line over a known length"),
        ("Drag an end", "Adjust it; right-click removes the line"),
    ]),
    ("Tools and history", [
        ("D / E / L", "Draw / Edit / Line tool"),
        ("Ctrl+Z", "Undo (also removes pending points)"),
        ("Ctrl+Shift+Z / Ctrl+Y", "Redo"),
        ("Enter", "On a finished zone: go to the next zone to do"),
    ]),
    ("View", [
        ("Ctrl + wheel", "Zoom at the cursor"),
        ("Middle-drag / Space + drag", "Pan"),
        ("F / Ctrl+0", "Fit the frame to the view"),
        ("1 / Ctrl+1", "Actual size (100%)"),
        ("? / F1", "Show or hide this help"),
    ]),
]


def help_html() -> str:
    rows = []
    for heading, entries in SHORTCUT_GROUPS:
        rows.append(f"<tr><td colspan='2'><b>{heading}</b></td></tr>")
        for keys, action in entries:
            rows.append(
                f"<tr><td style='padding-right:14px'><code>{keys}</code></td>"
                f"<td>{action}</td></tr>"
            )
    return (
        "<div style='font-size:11px'><b style='font-size:13px'>ROI editor "
        "shortcuts</b> (click, Esc or ? to close)<table cellspacing='1'>" + "".join(rows)
        + "</table></div>"
    )


class HelpOverlay(QLabel):
    """A translucent panel over the canvas listing the shortcuts."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setText(help_html())
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setStyleSheet(
            "background-color: rgba(20, 20, 20, 225); color: #f0f0f0; "
            "padding: 10px; border: 1px solid #888; border-radius: 6px;"
        )
        self.setWordWrap(False)
        self.hide()

    def place(self, area) -> None:
        """Centre the panel in ``area`` (the canvas viewport rectangle)."""
        hint = self.sizeHint()
        width = min(hint.width(), max(area.width() - 20, 100))
        height = min(hint.height(), max(area.height() - 20, 60))
        self.setGeometry(
            area.x() + (area.width() - width) // 2,
            area.y() + (area.height() - height) // 2,
            width, height,
        )

    def mousePressEvent(self, event):
        self.hide()
        event.accept()
