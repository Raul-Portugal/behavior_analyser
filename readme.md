# Behavioral Maze Analyzer

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](#)

A flexible, user-friendly desktop application for analyzing animal behavior in common laboratory mazes. This tool provides a complete graphical workflow from video import to data export, designed for researchers who need reliable, reproducible results without extensive programming knowledge.

> **Development status:** This is active research software. Always review the
> validation video and tracking coverage before interpreting results.
> Criterion validation of the automated scores against blinded manual scoring
> is still pending.
>
> **3.2.0 results differ from 3.1.x:** the detection tuner now matches the
> production analysis resolution, centroids are sub-pixel, Freestyle immobility
> is recomputed from smoothed speed (N/A when uncalibrated), the ROI
> bounding-box calibration fallback is gone, and TST uses engine v3. Do not
> pool 3.1.x and 3.2.0 results without re-analysing. See
> [CHANGELOG.md](CHANGELOG.md) for the complete list.
>
> **3.1 compatibility note:** TST configurations saved by earlier releases are
> deliberately rejected. TST 3.1 uses a new temporal rigid-motion engine, so
> each setup must be reviewed in the tuner and saved again.
>
> **3.2 compatibility note:** TST 3.2 uses engine
> `tst_motion_decomposition_v3`, which normalizes motion features by frame
> rate and analyses frames at a canonical 480 px width. TST configurations and
> tuner profiles saved by 3.1 (engine v2) are rejected with a re-tune message:
> open the video in the TST tuner, review the evidence and save a new
> configuration. Checkpoints from v2 runs cannot be resumed.

## Key Features

- **Standalone Application**: Run the software directly via a Windows executable—no Python setup required.
- **Batch Processing**: Analyze entire folders of videos in a single run with consistent or individually customized settings.
- **Multi-Maze Support**: Comes with built-in, highly optimized logic for:
  - **Y-Maze**: Calculates spontaneous alternation, arm entries, and same-arm returns (includes hysteresis filtering for overlapping center zones).
    The entry options dialog sets the minimum dwell time and whether the arm
    the animal starts in counts as an entry. Entries that a tracking gap may
    have hidden are reported as `entries_possibly_missed`.
  - **Elevated Plus Maze (EPM)**: Calculates time/distance in open/closed arms and preference percentages,
    with the same dwell and initial-arm options and `entries_possibly_missed`.
  - **Tail Suspension Test (TST)**: Separates articulated escape activity from
    coherent pendulum-like translation and rotation, calculates immobility and
    bouts, and supports multiple mice simultaneously.
  - **Freestyle / Open Field**: Custom zone drawing or zone-free tracking for pure kinematic analysis (speed, distance, transitions).
    Immobility is scored from smoothed speed against a cm/s threshold with a
    minimum bout length. The setup dialog exposes the immobility threshold
    (cm/s), the speed smoothing window (s), the minimum immobility bout (s)
    and the minimum zone dwell (s). Immobility is reported as N/A when the
    video is not calibrated.
- **Distance Calibration**: Always draw a reference line over something of known
  length, and enter its length in cm, to get distances and speeds in cm. Without
  it, distances are reported in pixels and Freestyle immobility is N/A. There is
  no automatic scale estimate from the ROI bounding box.
- **Interactive ROI Definition**: Visually draw polygons for each maze arm or testing zone directly on a video frame.
- **Interactive Detection Tuning**: Fine-tune detection against the production
  segmentation mask, selected-component overlay, and deterministic Quick Scan
  quality monitor.
- **Evidence-Based Tracking QC**: Foreground components are scored using area,
  contrast, dominance, and temporal continuity. Per-frame confidence and
  failure reasons are exported, and weak runs are explicitly marked for
  quality review.
- **Resumable Analysis**: Built-in checkpoint manager saves detector and TST
  temporal state so a resumed run is equivalent to uninterrupted processing.
- **Explicit Tracking Gaps**: Frames where the animal is not detected are
  exported as `undetected`; they are not counted as time outside the arena or
  as immobility.
- **Comprehensive Outputs**:
  - Frame-by-frame tracking data (CSV)
  - Detailed statistical summaries (CSV, JSON)
  - Batch summary CSV for easy group-level statistics.
  - Rich visualizations: Heatmaps, trajectory plots, ethograms, and fully annotated timelapse/validation videos.
  - A timestamped run folder with separate `data`, `plots`, and optional
    `media` directories per video, plus a machine-readable run manifest.

---

## Installation

### Option 1: For Standard Users (Recommended)

If you just want to use the software on a Windows machine, you do not need to install Python.

1. Go to the [Releases](../../releases) page of this repository.
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

For a reproducible release build, use `requirements-lock.txt`.

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
4. **Register the Maze**: Add your maze class constructor to the
   `AVAILABLE_MAZES` factory dictionary in `mazes/__init__.py`. The GUI will
   automatically incorporate it into the selection menus.

---

## Development Checks

Install the dependencies, then run the regression tests and the linter from
the repository root:

```bash
python -m unittest discover -s tests
python -m ruff check .
python gui_main.py --startup-check
```

The tests need the PyQt6 runtime. On a machine without a display, set
`QT_QPA_PLATFORM=offscreen` first. The full suite takes about a minute and a
half. `ruff` is configured in `pyproject.toml` (`pip install ruff`), and
`--startup-check` verifies that the application imports and starts, then exits
with status 0.

ROI templates are tied to the source video dimensions on which they were
drawn. The preflight check blocks reuse on videos with incompatible
dimensions; redraw the ROIs for those videos.

The detector tuner displays the actual production segmentation mask. The
selected component is green, and Quick Scan uses deterministic sequential
windows from the beginning, middle, and end of the recording.

The TST tuner reviews short start, middle and end sections without retaining
decoded video frames in memory. It uses the same engine, analysis geometry and
rate normalization as the production run, so thresholds tuned there carry over.
Production analysis still processes the complete selected interval. Purple means passive swing, red static immobility, green
active escape, and orange/yellow evidence is unscored and must be reviewed.

On Windows, `build_release.ps1` creates an isolated build environment, runs the
full test suite, and rebuilds `dist/Behavioral_Maze_Analyzer.exe`.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
