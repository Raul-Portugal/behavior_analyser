"""Shared helpers for dialogs that must remain usable on small displays."""
from PyQt6.QtWidgets import QApplication


def fit_dialog_to_available_screen(
    dialog,
    preferred_size=(1200, 800),
    minimum_size=(480, 360),
    margin=40,
):
    """Bound a dialog to the screen containing its parent/current window."""
    screen = dialog.screen() or QApplication.primaryScreen()
    if screen is None:
        dialog.setMinimumSize(*minimum_size)
        dialog.resize(*preferred_size)
        return

    available = screen.availableGeometry()
    maximum_width = max(320, available.width() - margin)
    maximum_height = max(240, available.height() - margin)
    dialog.setMinimumSize(
        min(minimum_size[0], maximum_width),
        min(minimum_size[1], maximum_height),
    )
    dialog.resize(
        min(preferred_size[0], maximum_width),
        min(preferred_size[1], maximum_height),
    )
