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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())