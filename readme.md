# Behavioral Maze Analyzer

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](#)

A flexible, user-friendly desktop application for analyzing animal behavior in common laboratory mazes (Y-maze, elevated plus maze, open field and tail suspension). Since 4.0 everything happens in one **workspace** window: a project holds your videos, their settings and the history of your runs, and you move between **Setup**, **Run** and **Results** without a chain of dialogs. It is designed for researchers who need reliable, reproducible results without extensive programming knowledge.

> **Development status:** This is active research software. Always review the
> validation video and tracking coverage before interpreting results.
> Criterion validation of the automated scores against blinded manual scoring
> must be performed per lab/setup, using the Validation menu and
> [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md), before results are
> published.
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

## What's new in 4.0

4.0.0 replaces the chain of modal dialogs with a single-window **workspace**:
a video table, the **Setup**, **Run** and **Results** tabs and a context panel
for the selected video. Your work is saved in a `.bmaproj` **project** (with
autosave and crash recovery) that also records every run. Setup steps can be
done in any order, with one template or per-video settings; the ROI editor is
embedded in the Setup tab; the Results tab opens a run, plays the validation
video, charts time bins by group and compares runs. **Analysis results are
unchanged from 3.5.0**, and a workspace run writes output files that are
byte-identical to the command line's. The classic interface is still there
(`--classic`). See [Getting started](#getting-started),
[OUTPUTS.md](OUTPUTS.md) and [CHANGELOG.md](CHANGELOG.md).

## What's new in 3.5

3.5.0 adds tools for criterion validation of the automated scores against
blinded manual scoring: a **Validation** menu with blind manual scoring,
a tracking position spot-check and a self-contained agreement report (ICC,
Bland-Altman, kappa, event precision and recall, pre-registrable acceptance
criteria), plus `VALIDATION_PROTOCOL.md`. **Analysis results are unchanged
from 3.4.0.** See [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md),
[OUTPUTS.md](OUTPUTS.md) and [CHANGELOG.md](CHANGELOG.md).

## What's new in 3.4

3.4.0 adds parallel batch processing, a headless command line, an open-field
arena analysis (center, periphery, thigmotaxis), sample-sheet metadata,
per-time-bin outputs and ROI template alignment. **Results for existing
analyses are unchanged compared with 3.3.0**; the new metrics and files appear
only when the new features are used, and parallel and sequential runs produce
byte-identical outputs. See [Running several videos at once](#running-several-videos-at-once),
[Command line](#command-line) and [CHANGELOG.md](CHANGELOG.md).

## What's new in 3.3

3.3.0 is a performance and usability release. **Analysis results are identical
to 3.2.0**, so 3.2.0 and 3.3.0 results can be pooled. The TST analysis is
about 4x faster with three mice, validation videos and plots are faster, and
the application now remembers your folders and settings, can resume an
interrupted run after a restart, and has keyboard shortcuts. See
[CHANGELOG.md](CHANGELOG.md) for the complete list.

## Getting started

1. **Create a project.** Start the application (`Behavioral_Maze_Analyzer.exe`,
   or `python gui_main.py`) and choose **File > New project...** (Ctrl+N), then
   name the `.bmaproj` file. To continue earlier work, use **File > Open
   project...** (Ctrl+O) or **File > Open recent**. If you used 3.x, the first
   start offers a welcome card, and **File > Import** brings in a saved
   template, a batch plan or a previous run folder.
2. **Add videos.** **Add videos...** (Ctrl+Shift+V), **Add folder...**
   (Ctrl+Shift+O), or drag videos and folders onto the window. The video table
   lets you edit metadata (**Add column**), search, filter by status and
   relink videos that moved. Choose the **Paradigm** in the panel on the right.
3. **Setup.** In the **Setup** tab, work through the steps in any order:
   paradigm parameters, regions of interest, calibration, detection (or TST
   motion) tuning, time window and review. Edit the **Project template** that
   every video follows, or switch to **Selected video(s)** to give some videos
   their own settings. **Apply** writes a step into the project.
4. **Run.** In the **Run** tab, pick the output folder and the scope (all
   configured videos, the selected ones, or the ones not yet done), then press
   **Run analysis** (F5). The pre-run checks, per-video progress, log, Cancel
   and **Resume interrupted run** are all there.
5. **Results.** The **Results** tab opens the finished run: the summary table,
   per-video metrics, plots and warnings, the validation video, time-bin charts
   and a comparison of two runs. Everything is also on disk in the
   `analysis_run_...` folder (see [OUTPUTS.md](OUTPUTS.md)).

[quickstart.md](quickstart.md) walks through all of this step by step.

### Classic interface

The 3.x wizard window (**Add Videos**, **Configure & Start Analysis**, ...) is
still available: start it with `python gui_main.py --classic`, or choose
**File > Switch to classic interface** in the workspace. It runs the same
analysis, so results are identical. **File > Export batch plan...** in the
workspace writes a 3.x batch plan that the classic interface and the command
line can load; `.bmaproj` projects cannot be opened by versions before 4.0.

## Key Features

- **Single-Window Workspace and Projects (4.0)**: A `.bmaproj` project keeps the videos, per-video metadata, settings, verification state and run history in one file, with autosave and recovery. The video table, the Setup / Run / Results tabs and the context panel replace the sequence of modal dialogs; imports from 3.x templates, batch plans and run folders are built in.

- **Standalone Application**: Run the software directly via a Windows executable (no Python setup required).
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
    With **Open field arena** enabled you draw the arena boundary first and
    choose how the center is defined: a distance from the wall in cm, or a
    share of the arena area. The run then also reports center time, center
    entries and latency, thigmotaxis (percentage of time and of distance),
    mean distance to the wall and center versus periphery speed, plus a
    per-frame `*_arena_zones.csv`. Detection is restricted to the arena.
- **Distance Calibration**: Always draw a reference line over something of known
  length, and enter its length in cm, to get distances and speeds in cm. Without
  it, distances are reported in pixels and Freestyle immobility is N/A. There is
  no automatic scale estimate from the ROI bounding box.
- **Interactive ROI Editor**: Draw polygons for each maze arm or testing zone on a zoomable video frame, in any order, then keep editing them: move vertices or whole polygons, insert and delete vertices, rotate and scale, undo and redo every edit, and snap shared borders. It is embedded in the Setup tab; in the classic interface, Redraw during verification reopens the editor on the existing ROIs.
- **Template Alignment**: Reuse ROIs on videos recorded with a slightly moved
  camera or at another resolution. *Tools > Align template to videos...*
  compares each video's background with the template's, flags the ones that no
  longer fit, and aligns them automatically or from landmarks you click; the
  reference line and cm calibration follow the alignment.
- **Parallel Batches and Command Line**: Analyze several videos at once with
  the **Parallel videos** control, or run a saved plan without the GUI with
  `python -m core.cli run` (see below).
- **Sample Sheet and Time Bins**: Attach animal metadata (`animal_id`, `group`,
  ...) from a CSV to the batch summary, and optionally report every metric in
  fixed-length time bins (`*_time_bins.csv`, `batch_time_bins.csv`).
- **Interactive Detection Tuning**: Fine-tune detection against the production
  segmentation mask, selected-component overlay, and deterministic Quick Scan
  quality monitor.
- **Evidence-Based Tracking QC**: Foreground components are scored using area,
  contrast, dominance, and temporal continuity. Per-frame confidence and
  failure reasons are exported, and weak runs are explicitly marked for
  quality review.
- **Resumable Analysis**: Built-in checkpoint manager saves detector and TST
  temporal state so a resumed run is equivalent to uninterrupted processing.
  Runs can be resumed even after closing the application (see
  [Resuming an interrupted run](#resuming-an-interrupted-run)).
- **Results Review in the App**: The Results tab shows the batch summary
  (sortable, filterable, exportable), per-video metrics, plots and quality
  warnings, plays the validation video (or the original video with tracking
  overlays), charts time bins per video or per group and compares two runs.
- **Explicit Tracking Gaps**: Frames where the animal is not detected are
  exported as `undetected`; they are not counted as time outside the arena or
  as immobility.
- **In-app Validation Tools** (Validation menu, in both interfaces): blind manual scoring with
  hotkeys on a built-in frame-accurate video player (Y-maze, EPM, open field,
  TST), a tracking position spot-check, and an agreement report (ICC,
  Bland-Altman, event precision/recall, kappa, acceptance criteria) written as
  a self-contained HTML file with CSV tables. See
  [VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md).
- **Comprehensive Outputs**:
  - Frame-by-frame tracking data (CSV)
  - Detailed statistical summaries (CSV, JSON)
  - Batch summary CSV for easy group-level statistics.
  - Rich visualizations: Heatmaps, trajectory plots, ethograms, and fully annotated timelapse/validation videos.
  - A timestamped run folder with separate `data`, `plots`, and optional
    `media` directories per video, plus a machine-readable run manifest.

---

## Resuming an interrupted run

Every run folder contains a `run_plan.json` with the exact settings of that
run. If a run was stopped (cancelled, crashed, or the application was
closed), its checkpoints stay in the output folder's `.checkpoints/` folder.
In the workspace, the **Run** tab shows **Resume interrupted run** when the
output folder holds checkpoints of project videos whose settings are
unchanged. In the classic interface, after a restart (at startup and whenever
videos are added), the application pairs each checkpoint with the run plan
whose settings match it and shows a **Resume interrupted run** banner (also
**File > Resume interrupted run...**). Either way the run continues from the
saved frame and the result is equivalent to an uninterrupted run. Checkpoints
older than 7 days, from another version, or that no run plan explains are
ignored.

To apply the settings of an earlier run to new videos, use **File > Import >
Previous run folder...** in the workspace (it creates a project from the run)
or **File > Load settings from previous run...** in the classic interface.

## Running several videos at once

The **Parallel videos** control (in the Run tab, or next to the Start button in
the classic interface) sets how many videos
are analysed at the same time. **Auto** (the default) uses the smaller of
CPU cores / 4, the number of videos and RAM in GB / 3, and at least 1; set 1 to
run the videos one after another. Each video runs in its own spawned process,
and the outputs are byte-identical to a sequential run (about 2.2x faster on a
3-video batch). The setting is remembered between sessions.

## Command line

A saved batch plan (**File > Export batch plan...** in the workspace,
**File > Save Configuration** in the classic interface, or the `run_plan.json`
of an earlier run) can be run without the GUI and without Qt:

```bash
python -m core.cli run --plan plan.json --out results     --workers 2 --sample-sheet sheet.csv --bin-seconds 60
```

It writes the same run folder, batch summary, manifest and `run_plan.json` as a
GUI run (byte-identical files; this is tested for the workspace). `--workers N` sets the parallel videos (default: Auto, as above),
`--sample-sheet` attaches a metadata CSV, `--bin-seconds N` turns on time-bin
outputs, `--quiet` prints only failures and the result. A template (instead of
a plan) needs `--videos` (files or folders), and `--paradigm` when it does not
name one. Exit codes: `0` success, `1` at least one video failed, `2` bad
arguments or plan, `130` cancelled. The packaged executable has a hidden
`--selftest-parallel` flag that checks that its process pool starts.

## Sample sheet and time bins

**Sample sheet...** (in the Run tab; or `--sample-sheet`) takes a CSV whose first
column is `video` and whose other columns are free metadata (`animal_id`,
`group`, ...). In the workspace the values also appear in the video table, and
you can add your own columns with **Add column**.
They are added to `batch_summary.csv`, each `*_summary.json` and the manifest;
videos are matched to rows, never guessed. **Time bins** (set in seconds in the Run tab, in minutes in the classic
interface, or `--bin-seconds`) also writes each metric per bin in long format; each metric summed over the
bins equals the total. Both are off by default and change nothing when off.
See [OUTPUTS.md](OUTPUTS.md) for the formats.

## Keyboard shortcuts

The workspace, and the editors and tuners it hosts:

| Where | Keys | Action |
|---|---|---|
| Workspace | Ctrl+N | New project |
| Workspace | Ctrl+O | Open a project (or import a template or batch plan) |
| Workspace | Ctrl+S | Save the project |
| Workspace | Ctrl+Shift+S | Save the project as... |
| Workspace | Ctrl+W | Close the project |
| Workspace | Ctrl+Q | Quit |
| Workspace | Ctrl+Shift+V | Add videos |
| Workspace | Ctrl+Shift+O | Add a folder of videos |
| Workspace | F5 | Run the analysis (Cancel is a toolbar and Run tab button; it has no key) |
| Video table | Del | Remove the selected videos from the project |
| Video table | Alt+Up, Alt+Down | Move the selected videos up or down |
| Video table | F2 or double-click | Edit a metadata cell |
| Results table | Ctrl+C | Copy the selected cells |
| Results video player | Space | Play or pause (when the player has focus) |
| Results video player | Left, Right | Step one frame |
| Results video player | Shift+Left, Shift+Right | Step ten frames |
| Results video player | Home, End | First or last frame |
| ROI editor | Click | Add a point to the polygon being drawn (snaps to nearby vertices) |
| ROI editor | Enter, double-click, right-click | Close the polygon; with none open, go to the next zone to do |
| ROI editor | Esc | Cancel the polygon in progress (in the classic dialog, Esc again closes the editor, asking first) |
| ROI editor | Backspace | Remove the last point |
| ROI editor | Ctrl+Z | Undo any edit (pending points first) |
| ROI editor | Ctrl+Shift+Z, Ctrl+Y | Redo |
| ROI editor | D, E, L | Draw, Edit and reference-Line tools |
| ROI editor | Delete, right-click | Delete the selected vertex or polygon (the line, in the Line tool) |
| ROI editor | Double-click an edge | Insert a vertex |
| ROI editor | Arrow keys (Shift: x10) | Nudge the selected vertex or polygon by 1 px |
| ROI editor | Alt (held) | Place or drag a point without snapping |
| ROI editor | Ctrl+wheel | Zoom at the cursor |
| ROI editor | Middle-drag, Space+drag | Pan |
| ROI editor | F or Ctrl+0, 1 or Ctrl+1 | Fit the frame, actual size (100%) |
| ROI editor | ? or F1 | Show or hide the shortcut help |
| TST tuner | Space | Play or pause |
| TST tuner | Left, Right | Step one frame |
| TST tuner | Ctrl+Left, Ctrl+Right | Step ten frames |
| Detection tuner | N, P | Next or previous frame |
| Detection tuner | + (or =), - | Forward or back 100 frames |

### Classic interface

| Where | Keys | Action |
|---|---|---|
| Main window | Ctrl+O | Add videos |
| Main window | Ctrl+Shift+O | Add a folder of videos |
| Main window | Ctrl+S | Save template |
| Main window | Ctrl+L | Load configuration or batch plan |
| Main window | F5 | Configure and start analysis |
| Main window | Esc | Cancel the running analysis (asks for confirmation) |
| Video list | Del or Backspace | Remove the selected videos |
| Video list | Alt+Up, Alt+Down | Move the selected videos up or down |

## Where settings are stored

The application remembers folders, window layout, the last paradigm, recent
files, recent projects, the project that was open last, each project's run
output folder and whether the welcome card was dismissed between sessions. These are stored per user with
Qt's QSettings, which on Windows is the registry (under the current user);
the application does not write them next to the executable. To use an ini file
instead (for example for a portable setup or to start from clean settings),
set the `BEHAVIORAL_MAZE_ANALYZER_SETTINGS_FILE` environment variable to the
path of that file before starting the application. Analysis settings
themselves are saved with each run in its `run_plan.json`, not in QSettings.

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
python gui_main.py                 # the workspace window
python gui_main.py study.bmaproj   # ... opening a project
python gui_main.py --classic       # the classic wizard window
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
`--startup-check` verifies that the application imports and starts (both the
workspace and the classic window), then exits with status 0.

ROI templates are tied to the source video dimensions on which they were
drawn. The preflight check blocks reuse on videos with incompatible
dimensions; use *Tools > Align template to videos...* to align the template to
those videos (or to a camera that has moved), or redraw the ROIs.

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
