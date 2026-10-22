---
paths: ["gui/**/*.py", "gui_main.py", "tools/ui_gallery.py"]
---
# Rules for Qt / GUI code (loaded when these files are touched)

- **Never call `QObject.sender()`.** Pass objects explicitly (lambda or functools.partial).
- **Timers:** create them as `QTimer(self)`. Never use a free `QTimer.singleShot(..., obj.method)` into an object that may be deleted.
- **Threads:** every `QThread` is interrupted and joined, and every `cv2.VideoCapture` is released, on hide, close and `shutdown()`. Follow `gui/tuners/lifecycle.py`.
- **Workspace dialogs** go through `window.prompts`. Tests replace it; a raw `QMessageBox.exec()` will hang the suite.
- **Styling:** use the design system (`gui/workspace/theme.py`, `ui_kit.py`). Set `set_class`/`set_role` properties, not ad-hoc per-widget stylesheets or hard-coded colours.
- **Labels:** plain-language labels with units. Every disabled control gets a tooltip saying why it is disabled.
- **Visual QA:** check every visible change by rendering with `tools/ui_gallery.py` and reading the PNGs. Do this at 1366×768 and 1920×1080, in light and dark.
- **Tests:** they run with `QT_QPA_PLATFORM=offscreen`. Offscreen fonts are fallback fonts, so don't assert pixel text widths.
