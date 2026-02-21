# Behavioral Maze Analyzer

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](#)

A flexible, user-friendly desktop application for analyzing animal behavior in common laboratory mazes. This tool provides a complete graphical workflow from video import to data export, designed for researchers who need reliable, reproducible results without extensive programming knowledge.

## Key Features

- **Standalone Application**: Run the software directly via a Windows executable, no Python setup required.
- **Batch Processing**: Analyze entire folders of videos in a single run with consistent or individually customized settings.
- **Multi-Maze Support**: Comes with built-in, highly optimized logic for:
  - **Y-Maze**: Calculates spontaneous alternation, arm entries, and same-arm returns (includes hysteresis filtering for overlapping center zones).
  - **Elevated Plus Maze (EPM)**: Calculates time/distance in open/closed arms and preference percentages.
  - **Tail Suspension Test (TST)**: Calculates motion energy, immobility time, and bouts for multiple mice simultaneously.
  - **Freestyle / Open Field**: Custom zone drawing or zone-free tracking for pure kinematic analysis (speed, distance, transitions).
- **Interactive ROI Definition**: Visually draw polygons for each maze arm or testing zone directly on a video frame.
- **Interactive Detection Tuning**: Fine-tune animal detection parameters with a live preview, confidence heatmaps, and a "Quick Scan" quality monitor.
- **Resumable Analysis**: Built-in checkpoint manager saves your progress. If your computer crashes or you cancel a batch, you can resume exactly where you left off.
- **Comprehensive Outputs**:
  - Frame-by-frame tracking data (CSV)
  - Detailed statistical summaries (CSV, JSON)
  - Batch summary CSV for easy group-level statistics.
  - Rich visualizations: Heatmaps, trajectory plots, ethograms, and fully annotated timelapse/validation videos.

---

## Installation

### Option 1: For Standard Users (Recommended)

If you just want to use the software on a Windows machine, you do not need to install Python.

1. Go to the [Releases](https://github.com/Raul-Portugal/behavior_analyser_v8/releases) page of this repository.
2. Download the latest `Behavioral_Maze_Analyzer.exe` file.
3. Double-click the `.exe` to run the application.

### Option 2: For Developers (Source Code)

If you want to modify the code, add new mazes, or run it on macOS/Linux:

**1. Clone the repository:**

```bash
git clone https://github.com/Raul-Portugal/behavior_analyser.git
cd behavior_analyser
```

**2. Set up a virtual environment (Recommended):**

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Run the application:**

```bash
python gui_main.py
```

---

## Extending the Software

The application uses a modular Core Architecture, making it easy to add new maze types. To add a new analysis module (e.g., Morris Water Maze):

1. **Create a Result Dataclass**: In `mazes/water_maze.py`, create a dataclass inheriting from `BaseAnalysisResult` for your specific metrics.
2. **Create a Maze Class**: Inherit from the abstract `Maze` class (in `mazes/base_maze.py`).
3. **Implement Abstract Methods**: Define `get_roi_definitions()`, `calculate_metrics()`, and any specific plotting functions.
4. **Register the Maze**: Add your new class to the `AVAILABLE_MAZES` dictionary in `mazes/__init__.py`. The GUI will automatically incorporate it into the selection menus.
