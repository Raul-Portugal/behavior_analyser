"""
Main entry point for the Behavioral Analysis Suite GUI application.
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

from gui.main_window import MainWindow

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


def main(arguments=None):
    """Launch the GUI or perform a packaged dependency/startup check."""
    arguments = list(sys.argv if arguments is None else arguments)
    startup_check = "--startup-check" in arguments
    qt_arguments = [
        argument for argument in arguments
        if argument != "--startup-check"
    ]
    sys.excepthook = _handle_uncaught_exception
    # Root-logger level is application configuration, so it lives here rather
    # than as a side effect of constructing the window.
    logging.getLogger().setLevel(logging.INFO)
    app = QApplication(qt_arguments)
    main_win = MainWindow()
    if startup_check:
        main_win.close()
        app.processEvents()
        return 0
    main_win.show()
    return app.exec()


if __name__ == '__main__':
    # Frozen builds start parallel-batch child processes by re-running this
    # executable; this call turns those children into pool workers.
    multiprocessing.freeze_support()
    sys.exit(main())
