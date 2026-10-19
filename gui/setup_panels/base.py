"""Common contract of the embeddable setup panels."""
from typing import Any, Dict, List, Tuple

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget


class SetupPanel(QWidget):
    """A non-modal setup step that edits a ``SetupState``.

    ``load(state)`` shows the state's values (without emitting ``changed``),
    ``apply(state)`` writes the panel's values back and marks the step done
    when nothing blocks it, ``issues()`` lists what currently prevents that
    and ``changed`` fires whenever the user edits a control.

    Panels never block and never open message boxes; dialogs that host a
    panel decide how to present ``issues()``.
    """

    changed = pyqtSignal()

    #: The ``Maze.setup_steps`` id this panel completes.
    step_id = ""

    def __init__(self, parent=None, preserve_exact: bool = True):
        super().__init__(parent)
        self._loading = False
        # With ``preserve_exact`` a loaded value the user never touched is
        # written back as loaded, not as the (rounded) number a spin box
        # shows. The classic dialogs switch it off to keep reading back what
        # the box shows, exactly as they always did.
        self._exact = ExactValues(preserve_exact)

    # -- contract ---------------------------------------------------------
    def load(self, state) -> None:
        raise NotImplementedError

    def apply(self, state) -> bool:
        """Write the panel into ``state``; False (and no change) if invalid."""
        raise NotImplementedError

    def issues(self) -> List[str]:
        return []

    # -- helpers for subclasses --------------------------------------------
    def _emit_changed(self, *_args) -> None:
        """Slot for control signals; silent while ``load`` is populating."""
        if not self._loading:
            self.changed.emit()

    def _finish_apply(self, state) -> bool:
        if self.issues():
            return False
        if self.step_id:
            state.mark_done(self.step_id)
        return True


class ExactValues:
    """Keeps a loaded value exact while its spin box shows a rounded one.

    A spin box rounds to its decimals, so showing 0.125 in a two-decimal box
    and reading it back would silently turn the parameter into 0.13. Values
    the user did not touch are therefore read back as loaded.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._seen: Dict[Any, Tuple[float, float]] = {}

    def set(self, spin, value) -> None:
        spin.setValue(value)
        self._seen[spin] = (value, spin.value())

    def get(self, spin):
        shown = spin.value()
        raw, rounded = self._seen.get(spin, (None, None))
        if self.enabled and raw is not None and shown == rounded:
            return raw
        return shown
