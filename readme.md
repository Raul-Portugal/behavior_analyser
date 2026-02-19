# Behavioral Maze Analyzer

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/GUI-PyQt6-brightgreen.svg)](https://riverbankcomputing.com/software/pyqt/)

A flexible, user-friendly desktop application for analyzing animal behavior in common laboratory mazes. This tool provides a complete graphical workflow from video import to data export, designed for researchers who need reliable, reproducible results without extensive programming knowledge.

## Key Features

- **Graphical User Interface**: A clean and intuitive interface built with PyQt6.
- **Batch Processing**: Analyze multiple videos in a single run with consistent settings.
- **Multi-Maze Support**: Comes with built-in support for:
  - **Y-Maze**: Calculates spontaneous alternation, arm entries, same-arm returns, and more.
  - **Elevated Plus Maze (EPM)**: Calculates time in open/closed arms, entries, and preference percentages.
- **Extensible Architecture**: Easily add support for new maze types (e.g., Open Field, Morris Water Maze) by creating a new maze class.
- **Interactive ROI Definition**: Visually draw polygons for each maze arm and center zone directly on a frame from your video.
- **Interactive Detection Tuning**: Fine-tune the animal detection parameters with a live preview to ensure tracking accuracy.
- **Comprehensive Outputs**: Generates a variety of output files for each video, including:
  - Frame-by-frame tracking data (CSV)
  - Detailed analysis summaries (CSV, JSON)
  - Sequence and entry details (CSV)
  - Batch summary CSV for easy group statistics.
- **Rich Visualizations**: Automatically creates plots and videos to visualize the results:
  - Heatmap of animal position
  - Trajectory plot colored by zone
  - Zone occupancy over time
  - Y-Maze specific sequence and transition matrix plots
  - Annotated timelapse video of the analysis.

## Technology Stack

- **Python 3**: The core programming language.
- **OpenCV**: For all video processing and computer vision tasks.
- **PyQt6**: For the graphical user interface.
- **NumPy / SciPy**: For numerical operations and scientific computing.
- **Matplotlib / Seaborn**: for generating high-quality plots and visualizations.
- **tqdm**: For displaying progress bars during long operations.

---

## Installation

Follow these steps to set up the application on your system.

### 1. Prerequisites

- **Python 3.9** or newer. You can download it from [python.org](https://www.python.org/).
- **Git** for cloning the repository. You can get it from [git-scm.com](https://git-scm.com/).

### 2. Clone the Repository

Open a terminal or command prompt and run the following command to download the source code:

```bash
git clone <your-repository-url>
cd behavior_analyzer_v3
```

### 3. Set Up a Virtual Environment (Recommended)

Using a virtual environment is highly recommended to keep project dependencies isolated.

**On Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

With your virtual environment activated, install all the required libraries using the requirements.txt file:

```bash
pip install -r requirements.txt
```

*Note: On some minimal Linux distributions, you may need to install the Tkinter library separately for the file dialogs to work, e.g., `sudo apt-get install python3-tk`*

## Usage

Once installed, launch the application by running the gui_main.py script from the project's root directory:

```bash
python gui_main.py
```

Follow the on-screen instructions as detailed in the [QUICKSTART.md](./QUICKSTART.md) guide.

## Extending the Software (For Developers)

The application is designed to be easily extended with new maze types. To add a new analysis module (e.g., for an Open Field test):

1. **Create a Result Dataclass**: In a new file like `mazes/open_field.py`, create a dataclass that inherits from `BaseAnalysisResult` and add any specific metrics you need.

2. **Create a Maze Class**: In the same file, create a class that inherits from `Maze` (from `mazes.base_maze`).

3. **Implement Abstract Methods**: You must implement all abstract methods defined in the `Maze` class:
   - `name()`: The display name of your maze.
   - `get_roi_definitions()`: The list of ROIs to be drawn.
   - `get_result_class()`: Return your new result dataclass type.
   - `calculate_metrics()`: The core logic for calculating your maze's specific metrics.
   - `get_batch_summary_headers()` and `get_batch_summary_row()`: Define the columns for the batch summary CSV.
   - `generate_specific_plots()`: Logic to create any plots unique to your maze.

4. **Register the Maze**: In `mazes/__init__.py`, import your new maze class and add an instance of it to the `AVAILABLE_MAZES` dictionary.

The GUI will automatically pick up your new maze type and incorporate it into the workflow.

## License

This project is licensed under the MIT License. See the LICENSE file for details.