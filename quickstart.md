# Quickstart Guide

This guide will walk you through a complete analysis workflow, from loading your first video to reviewing the generated heatmaps and data.

**What's new in 3.3:** faster analysis (results identical to 3.2.0), remembered
settings, resume after a restart and keyboard shortcuts. See `CHANGELOG.md`.

## 1. Launch the Application
- **If using the executable (.exe):** Double-click the downloaded file.
- **If using Python:** Open your terminal, activate your virtual environment, and run `python gui_main.py`.

## 2. Load Your Videos
1. Click the **"Add Videos"** or **"Add Folder"** button at the top of the main window.
2. Select the video files (.mp4, .avi, etc.) you want to analyze. 
   *Note: All output files (CSVs, plots, timelapses) will be saved in the same directory as the first video in your list, unless you change it via `Tools -> Set Output Directory`.*

## 3. Start the Configuration Workflow
Click the **"Configure & Start Analysis"** button. 

### Step A: Select Maze Type
A dialog will appear asking you to select the paradigm (e.g., "Y-Maze", "Elevated Plus Maze", "Tail Suspension Test", or "Freestyle / Open Field"). Select the appropriate option.

### Step B: Draw Regions of Interest (ROIs)
1. You will see a frame from the middle of your video.
2. Follow the prompt at the top left to draw the current zone (e.g., "Arm A").
3. **Left-click** to drop points and draw a polygon around the arm.
4. **Right-click** to close and finish the shape.
5. Click **"Next Zone"** and repeat until all zones are drawn.
6. *Calibration:* You will be asked to draw a reference line. Draw a line along an object of known length in the real world (e.g., a 30cm maze arm). This allows the software to output distances in centimeters instead of pixels. **Always draw a reference line when you want cm**: the software no longer estimates a scale from the ROI bounding box. The TST does not use calibration and skips this prompt.

### Step B2: Paradigm Options
- **Y-Maze / Elevated Plus Maze:** after the ROIs, set the **Minimum dwell time** (a zone change must last this long to count, which ignores flicker across a zone border) and whether **the arm the animal starts in counts as an entry**.
- **Freestyle / Open Field:** the setup dialog also holds the immobility parameters: the **Immobility threshold** (cm/s), the **Speed smoothing window** (s), the **Minimum immobility bout** (s) and the **Minimum zone dwell** (s). Immobility is scored from the smoothed speed, and is reported as N/A if the video is not calibrated.

### Step C: Tune Detection Parameters
The Interactive Detection Tuner ensures the software cleanly separates the animal from the background.
1. Use the **Frame Navigation** slider (or the `N` and `P` keys) to look at different parts of the video.
2. Adjust the **Threshold**, minimum component area, and morphology controls
   until the green selected component in the "Threshold View" covers the
   animal without selecting reflections or shadows.
3. Check the evidence-based confidence and quality panel. “Review” or “Poor”
   means the validation output must be inspected and the settings may need
   adjustment.
4. *(Optional)* Click **Quick Scan Test** to test deterministic sequential
   windows from the beginning, middle, and end of the video.
5. Click **Accept Parameters**.

For TST, the evidence tuner is different: select each mouse in turn and review
the state overlay across the scan. Purple is coherent passive swing, red is
static immobility, green is active escape, and orange/yellow is uncertain and
excluded from scoring. The tuner reviews up to five minutes to remain
responsive; the actual analysis uses the entire selected interval.

### Step D: Finalize Time & Settings
The final review dialog allows you to:
- Adjust the **Start Time** and **End Time** (useful for trimming setup/handling time).
- Enter the real-world length of the calibration line you drew earlier.
- Check the box to generate a **Validation/Timelapse Video** (highly recommended for verifying results).

## 4. Run the Batch Analysis
- If you have multiple videos loaded, the software will ask if you want to **Review each video** or **Run all now**.
  - **Review each video:** Allows you to adjust start/end times or redraw ROIs slightly if the camera moved between recordings.
  - **Run all now:** Applies the exact same ROIs and settings to every video in the list.
- A single summary box lists the settings and any pre-flight warnings. Click **Start analysis** to begin. The progress bar shows the estimated time remaining for the batch.
- *Note: If you need to stop, click **Cancel Analysis** (or press Esc) and confirm. The software will save a checkpoint, and you can resume from the exact same frame later!*

### Animal metadata and time bins (optional)
Under the Start button, **Sample sheet...** attaches animal information to the
results and **Time bins** also reports the metrics in consecutive bins of N
minutes (Off by default; see `OUTPUTS.md`). A sample sheet is a CSV whose first
column is `video` (a bare filename, or a path relative to the sheet) followed by
any columns you like:

```text
video,animal_id,group,sex
mouse1.mp4,A01,control,F
mouse2.mp4,A02,treated,M
```

Those columns appear in `batch_summary.csv` (after the shared columns), in each
video's `*_summary.json` and in `manifest.json`. The sheet is checked when you
load it: videos with no row, or whose filename matches several rows, are
listed, and you are asked before starting with blank metadata for them.
Headless: `python -m core.cli run --plan plan.json --out DIR --sample-sheet
sheet.csv --bin-seconds 60`.

### Resume after a restart
Every run folder contains a `run_plan.json` with the exact settings of the
run. If a run was interrupted, even by closing the application, reopen the
application (or add the same videos again): a **Resume interrupted
run** banner (also **File > Resume interrupted run...**) reloads the matching
plan and continues from the saved frame. **File > Load settings from previous
run...** reuses the settings of an earlier run for new videos.

When reusing a saved configuration, the target videos must have the same frame
dimensions as the video used to draw the ROIs. The preflight check will stop
the run if the geometry is incompatible.

## 5. Review Your Results
Once complete, open the new `analysis_run_...` folder inside your chosen
output directory. It contains `batch_summary.csv`, a `manifest.json` run index,
and one folder per video. Each video folder separates:

1. `data/`: frame-level data, human-readable summary, complete JSON, and any
   paradigm-specific event tables.
2. `plots/`: heatmaps, trajectories, time series, and paradigm-specific plots.
3. `media/`: optional annotated validation video; this folder is omitted when
   validation media was not requested.

The batch summary always includes the shared duration, total-distance/unit,
excluded-jump, and tracking-quality columns before paradigm-specific metrics.
Unsupported fields, such as centroid distance for TST, are left blank rather
than reported as zero. See `OUTPUTS.md` for the full per-paradigm output
matrix.

In tracking CSVs, `zone=undetected` means no reliable position was available
for that frame. These frames are excluded from zone occupancy, distance,
transition, and immobility calculations. The same file includes `confidence`
and `failure_reason` columns for quality review.

Configurations saved by releases before 3.2 cannot be used for TST. Re-open
the source video, review every mouse with the temporal evidence tuner, and
save a new configuration.

Results from 3.2.0 are not directly comparable with 3.1.x; see
`CHANGELOG.md`. Results from 3.3.0 are identical to 3.2.0 and can be pooled
with them. Criterion validation of the automated scores against blinded
manual scoring is still pending, so review the validation video before
interpreting results.

## 6. Keyboard Shortcuts and Saved Settings

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
| ROI selector | Enter | Next |
| ROI selector | Backspace or Ctrl+Z | Undo the last point (reopens a closed shape) |
| TST tuner | Space | Play or pause |
| TST tuner | Left, Right | Step one frame |
| TST tuner | Ctrl+Left, Ctrl+Right | Step ten frames |
| Detection tuner | N, P | Next or previous frame |
| Detection tuner | + (or =), - | Forward or back 100 frames |

The application remembers folders, window layout, the last paradigm, recent
files and the output folder. They are stored in your user profile with Qt's
QSettings (the registry on Windows), not next to the executable. Set the
`BEHAVIORAL_MAZE_ANALYZER_SETTINGS_FILE` environment variable to the path of
an ini file to store them there instead.

## 7. Developer Checks
If you are running from source, check an installation or a change with:

```bash
python -m unittest discover -s tests
python -m ruff check .
```

Set `QT_QPA_PLATFORM=offscreen` for the tests on a machine without a display.
