"""
gui/tuners/lifecycle.py
Shared "background QThread + video capture + shutdown" handling for the
embeddable tuner widgets.

Why this exists
---------------
The tuners own heavy resources: a worker ``QThread`` that scans the video, and
one or more ``cv2.VideoCapture`` handles.  A modal ``QDialog`` could release
them in ``done()``.  An embedded widget never gets ``done()``: it is hidden
when the user switches step, closed with the main window, or deleted together
with its parent.  Destroying a ``QThread`` that is still running aborts the
whole process, so every way of leaving the screen must interrupt and join the
thread first.

The pattern (for the workspace author)
--------------------------------------
Every tuner widget owns one :class:`TunerLifecycle` and routes everything
through it::

    self._lifecycle = TunerLifecycle("detection tuner")
    self._lifecycle.attach(self)              # joins on C++ deletion too
    self._capture = self._lifecycle.open_capture(path)   # lazy, reopenable
    self._lifecycle.start_thread(worker, cancel=worker.cancel)

and implements three hooks with a fixed meaning:

``hideEvent`` (not spontaneous)
    ``lifecycle.pause()``: interrupt and join the threads, release the
    captures.  The widget stays usable: captures reopen lazily on next use and
    ``showEvent`` restarts timers / re-runs an interrupted scan.  This is what
    happens when the workspace switches to another setup step.
``shutdown()`` (also called by ``closeEvent``)
    ``lifecycle.shutdown()``: pause, then mark the lifecycle terminal.  Nothing
    reopens afterwards.  The workspace must call it when it discards the step
    (and a dialog wrapper calls it from ``done()``).  Idempotent.
``destroyed`` (installed by :meth:`TunerLifecycle.attach`)
    Last-resort safety net when the widget is deleted without ``shutdown()``
    (for example its parent was deleted mid-scan).

Rules that keep this safe
-------------------------
* Never give a worker thread the widget as its Qt parent.  ``QWidget``'s
  destructor deletes its children *before* ``destroyed`` fires, so a running
  child thread would be destroyed first and abort the process.  The lifecycle
  owns the threads instead (parentless, strong Python reference).
* Worker code must poll ``isInterruptionRequested()`` often and release the
  resources it opened itself in a ``finally`` block.
* Objects registered with the lifecycle must not reference the widget
  (the ``destroyed`` handler runs while the widget is being torn down and must
  only touch threads and captures).
* Connect worker signals to widget slots as bound methods of the widget and
  never use ``QObject.sender()``; late queued signals to a deleted widget are
  dropped by Qt, late signals to a live but shut-down widget should be
  ignored via ``lifecycle.is_shut_down``.
* A thread that refuses to stop within its timeout is kept alive in a module
  registry until it finishes, so it can never be destroyed while running.
"""
import logging
from typing import Callable, List, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)

# Threads that did not stop within their timeout.  Keeping a reference stops
# Python from destroying a running QThread (which aborts the process).
_STRAGGLERS: List[object] = []


def _is_running(thread) -> bool:
    try:
        return bool(thread.isRunning())
    except RuntimeError:  # the C++ object is already gone
        return False


def _retain_until_finished(thread) -> None:
    if thread in _STRAGGLERS:
        return
    _STRAGGLERS.append(thread)

    def release(_thread=thread):
        try:
            _STRAGGLERS.remove(_thread)
        except ValueError:
            pass

    thread.finished.connect(release)


def stop_thread(thread, cancel: Optional[Callable[[], None]] = None,
                timeout_ms: Optional[int] = None,
                label: str = "worker") -> bool:
    """Interrupt ``thread`` and wait for it; True when it is not running.

    ``cancel`` replaces ``requestInterruption`` for workers with their own
    cancellation method.  ``timeout_ms=None`` joins without a limit.
    """
    if thread is None or not _is_running(thread):
        return True
    if cancel is not None:
        cancel()
    else:
        thread.requestInterruption()
    stopped = thread.wait() if timeout_ms is None else thread.wait(timeout_ms)
    if not stopped:
        logger.warning("%s did not stop within %s ms", label, timeout_ms)
        _retain_until_finished(thread)
    return bool(stopped)


class CaptureHolder:
    """A ``cv2.VideoCapture`` that can be released and lazily reopened.

    ``get()`` opens the capture on first use and reopens it after
    ``release()``.  After the owning lifecycle is shut down it never reopens
    again and hands back the released capture, so late callers see
    ``isOpened() == False`` instead of leaking a new handle.
    """

    def __init__(self, path, factory: Callable = cv2.VideoCapture,
                 lifecycle: Optional["TunerLifecycle"] = None):
        self._path = str(path)
        self._factory = factory
        self._lifecycle = lifecycle
        self._capture = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None and bool(self._capture.isOpened())

    def get(self):
        closed = self._lifecycle is not None and self._lifecycle.is_shut_down
        if self._capture is None or (
            not closed and not self._capture.isOpened()
        ):
            self._capture = self._factory(self._path)
            if closed:
                # Shut down before ever being used: hand out a dead handle.
                self._capture.release()
        return self._capture

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()


class TunerLifecycle:
    """Owns a tuner's worker threads and captures; see the module docstring."""

    def __init__(self, label: str = "tuner"):
        self.label = label
        self._threads: List[
            Tuple[object, Optional[Callable[[], None]], Optional[int]]
        ] = []
        self._captures: List[CaptureHolder] = []
        self._shut_down = False

    @property
    def is_shut_down(self) -> bool:
        return self._shut_down

    def attach(self, widget) -> None:
        """Join threads and free captures if ``widget`` is deleted un-shut.

        The handler captures this lifecycle (not the widget), so it is valid
        while Qt tears the widget down.
        """
        widget.destroyed.connect(
            lambda *_args, lifecycle=self: lifecycle.shutdown()
        )

    def open_capture(self, path, factory: Callable = cv2.VideoCapture,
                     open_now: bool = True) -> CaptureHolder:
        holder = CaptureHolder(path, factory, self)
        self._captures.append(holder)
        if open_now:
            holder.get()
        return holder

    def start_thread(self, thread,
                     cancel: Optional[Callable[[], None]] = None,
                     timeout_ms: Optional[int] = None) -> bool:
        """Track ``thread`` (parentless!) and start it.

        Refuses (returns False) once shut down.  ``cancel`` / ``timeout_ms``
        are used by :meth:`stop_threads`.
        """
        if self._shut_down:
            return False
        self._forget_finished()
        self._threads.append((thread, cancel, timeout_ms))
        thread.start()
        return True

    def has_running_thread(self) -> bool:
        return any(_is_running(entry[0]) for entry in self._threads)

    def stop_threads(self) -> bool:
        """Interrupt and join every tracked thread; True if all stopped."""
        all_stopped = True
        for thread, cancel, timeout_ms in self._threads:
            all_stopped &= stop_thread(thread, cancel, timeout_ms, self.label)
        self._forget_finished()
        return all_stopped

    def release_captures(self) -> None:
        for holder in self._captures:
            holder.release()

    def pause(self) -> None:
        """Stop threads and release captures; the tuner can resume later."""
        self.stop_threads()
        self.release_captures()

    def shutdown(self) -> None:
        """Terminal pause: nothing is started or reopened afterwards."""
        if self._shut_down:
            return
        self._shut_down = True
        self.pause()

    def _forget_finished(self) -> None:
        self._threads = [
            entry for entry in self._threads if _is_running(entry[0])
        ]
