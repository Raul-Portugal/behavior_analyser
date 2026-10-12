"""
gui/background_task.py
Run a slow, blocking function off the UI thread behind a modal progress
dialog with a Cancel button.
"""
import logging
import threading
from typing import Any, Callable, Optional

from PyQt6.QtCore import QEventLoop, QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QProgressDialog, QPushButton, QWidget

logger = logging.getLogger(__name__)


class TaskCancelled(Exception):
    """The user cancelled a task started with :func:`run_with_progress`."""


class _TaskRunner(QObject):
    done = pyqtSignal(object, object)  # (result, exception)

    def __init__(self, fn: Callable[[Callable[[], bool]], Any],
                 should_cancel: Callable[[], bool]):
        super().__init__()
        self._fn = fn
        self._should_cancel = should_cancel

    def run(self):
        try:
            self.done.emit(self._fn(self._should_cancel), None)
        except BaseException as exc:  # handed to the caller's thread
            self.done.emit(None, exc)


class _Receiver(QObject):
    """Lives on the calling (UI) thread so ``done`` is delivered there."""

    def __init__(self, loop: QEventLoop):
        super().__init__()
        self._loop = loop
        self.outcome = {}

    @pyqtSlot(object, object)
    def on_done(self, result, error):
        self.outcome["result"] = result
        self.outcome["error"] = error
        self._loop.quit()


def run_with_progress(
    parent: Optional[QWidget],
    label: str,
    fn: Callable[[Callable[[], bool]], Any],
    title: str = "Please Wait",
) -> Any:
    """Run ``fn(should_cancel)`` in a worker thread and return its result.

    The UI stays responsive (a modal dialog blocks input to the parent only).
    ``fn`` should poll ``should_cancel()`` and may raise ``InterruptedError``
    to stop early.  Raises :class:`TaskCancelled` if the user pressed Cancel
    (even if ``fn`` happened to finish first) and re-raises any exception
    ``fn`` raised.
    """
    cancel_event = threading.Event()

    dialog = QProgressDialog(label, "Cancel", 0, 0, parent)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)

    def request_cancel():
        cancel_event.set()
        dialog.setLabelText("Cancelling...")
        for button in dialog.findChildren(QPushButton):
            button.setEnabled(False)

    # Replace the default slot, which would hide the dialog at once; it
    # stays up until the task has actually stopped.
    dialog.canceled.disconnect()
    dialog.canceled.connect(request_cancel)

    thread = QThread()
    runner = _TaskRunner(fn, cancel_event.is_set)
    runner.moveToThread(thread)
    loop = QEventLoop()
    receiver = _Receiver(loop)
    outcome = receiver.outcome

    thread.started.connect(runner.run)
    runner.done.connect(receiver.on_done, Qt.ConnectionType.QueuedConnection)
    thread.finished.connect(runner.deleteLater)

    thread.start()
    dialog.show()
    try:
        loop.exec()
        # Closing the dialog below also emits ``canceled``, so decide now.
        cancelled = cancel_event.is_set()
    finally:
        dialog.canceled.disconnect()
        thread.quit()
        thread.wait()
        dialog.close()
        dialog.deleteLater()
        thread.deleteLater()

    error = outcome.get("error")
    if isinstance(error, InterruptedError) or (
        error is None and cancelled
    ):
        raise TaskCancelled()
    if error is not None:
        raise error
    return outcome.get("result")
