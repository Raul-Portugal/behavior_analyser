"""
Main entry point for the Behavioral Analysis Suite GUI application.

``gui_main.py [project.bmaproj]`` opens the 4.0 workspace window (and the
project given on the command line); ``--classic`` opens the classic wizard
window instead; ``--startup-check`` builds and closes both and exits.
"""
import logging
import multiprocessing
import sys
import traceback
from pathlib import Path

# Add the project root to the Python path to ensure local modules can be imported.
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication, QMessageBox

from gui.app_settings import AppSettings
from gui.main_window import MainWindow
from gui.workspace import theme
from gui.workspace.window import WorkspaceWindow

logger = logging.getLogger(__name__)


def _handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Report unexpected errors instead of letting PyQt6 abort the process.

    PyQt6 calls qFatal() when an exception escapes a slot unless a custom
    excepthook is installed, which would terminate the application and lose
    any unsaved configuration.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical(
        "Unhandled exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    app = QApplication.instance()
    if app is None:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    details = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle("Unexpected Error")
    box.setText(
        f"An unexpected error occurred: {exc_value}\n\n"
        "The application will keep running, but the last action may not "
        "have completed. Please review the log before continuing."
    )
    box.setDetailedText(details)
    box.exec()


def _apply_workspace_theme(app):
    """Give the workspace its look (the classic window keeps the platform's).

    The mode is the user's saved choice ("auto" follows the OS light/dark
    setting). A theme problem must never keep the application from opening.
    """
    try:
        theme.apply_theme(app, AppSettings().theme_mode())
    except Exception:
        logger.warning("The workspace theme could not be applied", exc_info=True)


def main(arguments=None):
    """Launch the GUI or perform a packaged dependency/startup check."""
    arguments = list(sys.argv if arguments is None else arguments)
    if "--selftest-parallel" in arguments:
        # Hidden packaged-build check: runs a tiny parallel batch (no GUI);
        # an optional following argument is a file to write the report to.
        from core.selftest import run_parallel_selftest
        position = arguments.index("--selftest-parallel")
        report = (
            Path(arguments[position + 1])
            if len(arguments) > position + 1 else None
        )
        return run_parallel_selftest(report)
    startup_check = "--startup-check" in arguments
    classic = "--classic" in arguments
    flags = {"--startup-check", "--classic"}
    qt_arguments = [argument for argument in arguments if argument not in flags]
    project_argument = _path_argument(qt_arguments[1:])
    sys.excepthook = _handle_uncaught_exception
    # Root-logger level is application configuration, so it lives here rather
    # than as a side effect of constructing the window.
    logging.getLogger().setLevel(logging.INFO)
    app = QApplication(qt_arguments)
    if startup_check:
        # Both interfaces must construct and close cleanly in a packaged build
        # (the workspace themed, the classic window with the platform look).
        _apply_workspace_theme(app)
        for window_class in (WorkspaceWindow, MainWindow):
            window = window_class()
            window.close()
            app.processEvents()
            theme.reset_theme(app)
        return 0
    if classic:
        main_win = MainWindow()
        main_win.show()
        return app.exec()
    _apply_workspace_theme(app)
    workspace = WorkspaceWindow()
    workspace.show()
    workspace.schedule_start(project_argument)
    return app.exec()


def _path_argument(arguments):
    """The first command-line argument that names an existing file."""
    for argument in arguments:
        if argument.startswith("-"):
            continue
        candidate = Path(argument)
        if candidate.is_file():
            return candidate
    return None


if __name__ == '__main__':
    # Frozen builds start parallel-batch child processes by re-running this
    # executable; this call turns those children into pool workers.
    multiprocessing.freeze_support()
    sys.exit(main())
