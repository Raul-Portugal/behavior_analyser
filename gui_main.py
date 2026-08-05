"""
Main entry point for the Behavioral Analysis Suite GUI application.
"""
import sys
from pathlib import Path

# Add the project root to the Python path to ensure local modules can be imported.
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main(arguments=None):
    """Launch the GUI or perform a packaged dependency/startup check."""
    arguments = list(sys.argv if arguments is None else arguments)
    startup_check = "--startup-check" in arguments
    qt_arguments = [
        argument for argument in arguments
        if argument != "--startup-check"
    ]
    app = QApplication(qt_arguments)
    main_win = MainWindow()
    if startup_check:
        main_win.close()
        app.processEvents()
        return 0
    main_win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
