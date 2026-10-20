# Quickstart Guide

This guide will walk you through a complete analysis workflow, from loading your first video to reviewing the generated heatmaps and data.

**What's new in 3.5:** a **Validation** menu to check the automated scores
against your own blinded manual scoring (blind manual scoring, a tracking
spot-check and an agreement report); see section 6 and
`VALIDATION_PROTOCOL.md`. Results for existing analyses are unchanged from
3.4.0.

**What's new in 3.4:** parallel videos, a headless command line, an open-field
arena analysis, sample-sheet metadata, time bins and ROI template alignment.
Results for existing analyses are unchanged compared with 3.3.0. See
`CHANGELOG.md`.

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

### Step B: Draw and Edit Regions of Interest (ROIs)
The **ROI editor** opens on a frame from the middle of your video. The **Zones**
list on the left has one row per zone (the arena first, if it is switched on),
and a **Reference line** row at the bottom. Each row shows whether the zone is
done (✓), missing (○) or breaks a rule (✗, for example a TST mouse with two
polygons, or a polygon smaller than 100 px²). Click any row at any time to draw
or edit that zone: there is no fixed order.

**Drawing a zone** (the *Draw* tool, key `D`)
1. Pick the zone in the list (the editor starts on the first missing one).
2. **Click** to drop points around the arm.
3. Press **Enter**, **double-click** or **right-click** to close the polygon.
   **Esc** cancels the shape in progress; **Backspace** removes its last point.
   A zone that takes exactly one polygon (arena, TST mouse) moves you on to the
   next missing zone by itself.
4. New points **snap to existing vertices** within 6 screen pixels, so
   neighbouring zones (for example the Y-maze arms and the center) can share an
   exact border. Hold **Alt** to place a point without snapping.

**Editing** (the *Edit* tool, key `E`): nothing has to be redrawn from scratch.
- Drag a **vertex** to move it, or drag a **polygon** to move all of it.
- **Double-click an edge** to insert a vertex. Press **Delete** or
  **right-click** to remove the selected vertex or polygon (a polygon keeps at
  least three vertices).
- With a polygon selected, the blue handle above it **rotates** and the orange
  handle at its corner **scales** it. The arrow keys nudge the selection by one
  pixel (Shift: ten).
- **Ctrl+Z** undoes and **Ctrl+Shift+Z** (or **Ctrl+Y**) redoes every edit, not
  just the last point. **Clear zone** removes all polygons of the selected zone
  in one undoable step.

**Zoom and pan:** **Ctrl+wheel** zooms at the cursor, **Fit** (or `F`) shows the
whole frame, `1` shows actual size, and a **middle-button drag** or
**Space+drag** pans. The status bar shows the cursor position in source pixels
and the zoom. Press **?** for the list of shortcuts.

**Calibration** (the *Line* tool, key `L`, or the Reference line row): draw a
line along an object of known length in the real world (e.g., a 30cm maze arm):
click two points or drag, then drag an end to adjust it; its length in pixels is
shown on the line. This allows the software to output distances in centimeters
instead of pixels. **Always draw a reference line when you want cm**: the
software no longer estimates a scale from the ROI bounding box. When you press
**OK** without a line you are asked once whether to draw one. The TST does not
use calibration and skips this prompt.

**Finish** with **OK** once every zone is done (a pending shape of three or more
points is closed for you); the editor lists the zones that still need work.
**Cancel** asks before it throws away changes.

### Step B2: Paradigm Options
- **Y-Maze / Elevated Plus Maze:** after the ROIs, set the **Minimum dwell time** (a zone change must last this long to count, which ignores flicker across a zone border) and whether **the arm the animal starts in counts as an entry**.
- **Freestyle / Open Field:** the setup dialog also holds the immobility parameters: the **Immobility threshold** (cm/s), the **Speed smoothing window** (s), the **Minimum immobility bout** (s) and the **Minimum zone dwell** (s). Immobility is scored from the smoothed speed, and is reported as N/A if the video is not calibrated.

- **Open-field arena (Freestyle):** tick **Open field arena** in the setup
  dialog to get center/periphery metrics and thigmotaxis. The ROI editor then
  lists the **arena boundary** first (the floor outline, drawn like any zone,
  before the zones and the reference line). Choose how the center is defined:
  a **distance from the wall in cm** (default 10 cm, needs the reference line)
  or a **share of the arena area** (default 25%; used automatically when the
  video is not calibrated). The ROI editor previews an area-share center.
  The run reports center time, center entries, latency to center, thigmotaxis
  (% of time and of distance), mean distance to the wall and center versus
  periphery speed, and writes `*_arena_zones.csv`. Detection is restricted to
  the arena, and the center outline is drawn in the validation video. Zones are
  optional in this mode; `arena` is a reserved zone name. See `OUTPUTS.md`.

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

### Workspace: Setup
In the workspace window the steps above live in the **Setup** tab, so nothing
pops up in front of the video list. The list on the left shows the steps of
the chosen paradigm (parameters, regions, calibration, detection or TST
tuning, time window, review); a green tick means done, a grey ring not done
yet and a red mark a problem, which is listed under the step. Steps can be
opened in any order; **Previous step** and **Next step** walk the list.

- **What you are editing** is shown at the top. *Project template* is the
  shared setting every video follows; changing it updates all the videos that
  use it. Switch the scope to *Selected video(s)* to give the videos selected
  in the table their own settings, starting from the template's values. A
  video that is edited this way stops following the template until you use
  **Apply template** on it again. With several videos selected, each step you
  apply is copied to all of them, and everything else of each video stays as
  it was.
- **Apply** writes the open step into the project. Nothing changes until you
  apply; leaving the step, switching scope or selection, closing the project
  or the window with unapplied edits asks whether to apply, discard or stay.
  **Revert step** drops the edits of the open step.
- The **regions** and the tuners work on a frame of the *reference video*:
  the first video of the project (pick another in the *Reference video* box)
  for the template, or the selected video itself. If a selected video's frame
  size differs from the one the template was drawn on, a warning with
  **Align template...** appears; the same dialog opens from **Tools >
  Align template to videos...**, which aligns the selected videos (or every
  video that does not fit) and stores the result as their own, "aligned"
  settings.
- **Draw reference line...** in the calibration step jumps to the regions step
  with the line tool selected.
- **Mark selected videos verified** records that you checked their regions;
  the review step also lists the problems found in the project's videos.

The settings built here are identical to the ones the classic wizard produces
for the same choices.

## 4. Run the Batch Analysis
- If you have multiple videos loaded, the software will ask if you want to **Review each video** or **Run all now**.
  - **Review each video:** Allows you to adjust start/end times or **Redraw ROIs** if the camera moved between recordings: the ROI editor opens with the current ROIs, arena and reference line already loaded, so you only move what changed.
  - **Run all now:** Applies the exact same ROIs and settings to every video in the list.
- A single summary box lists the settings and any pre-flight warnings. Click **Start analysis** to begin. The progress bar shows the estimated time remaining for the batch.
- *Note: If you need to stop, click **Cancel Analysis** (or press Esc) and confirm. The software will save a checkpoint, and you can resume from the exact same frame later!*

### Parallel videos
Next to the Start button, **Parallel videos** sets how many videos are analysed
at the same time. **Auto** (the default) picks `min(CPU cores / 4, videos,
RAM in GB / 3)`, at least 1; 1 runs the videos one after another. Outputs are
byte-identical to a sequential run.

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

### Running from the command line
A saved batch plan (or the `run_plan.json` of an earlier run) can be analysed
without the GUI, for example on a server:

```bash
python -m core.cli run --plan plan.json --out results --workers 2
```

The run folder is created inside `results` and is identical to a GUI run.
Options: `--workers N` (default Auto), `--sample-sheet`, `--bin-seconds`,
`--quiet`, and `--videos` (plus `--paradigm`) when you pass a template instead of
a plan. Exit codes: `0` success, `1` a video failed, `2` bad arguments or
plan, `130` cancelled.

### Resume after a restart
Every run folder contains a `run_plan.json` with the exact settings of the
run. If a run was interrupted, even by closing the application, reopen the
application (or add the same videos again): a **Resume interrupted
run** banner (also **File > Resume interrupted run...**) reloads the matching
plan and continues from the saved frame. **File > Load settings from previous
run...** reuses the settings of an earlier run for new videos.

### Reusing ROIs when the camera moved or the resolution differs

A saved template or plan holds ROIs in the pixels of the video they were drawn
on, so a video with another frame size, or one recorded with a slightly moved
camera, would get misplaced regions. The pre-flight check still stops a run
whose frame size does not match, but you can now **align** the template instead
of redrawing it:

1. Load the template (**File > Load Configuration / Plan...**) and add the videos.
2. Choose **Tools > Align template to videos...**. The application asks which
   video the template was drawn on (it remembers it for the session), compares
   each video's animal-free background with that video's, and lists the videos
   that need alignment: a different frame size, or a camera that moved so the
   ROIs would be off by more than 3 px. Videos that already fit are left alone.
3. For each flagged video the alignment window shows the template frame with
   its ROIs (left) and the video's frame with the transformed ROIs (right).
   - **Auto-align** matches the arena in the two background frames. Check the
     quality line: "good: 0.8 px" can be accepted; "check carefully: 6 px"
     means the match is doubtful.
   - **Pick landmarks...** lets you click 3-4 fixed points (arena corners,
     screws) on the template frame and then the same points on the video, in
     the same order. The fit and its residual update after each pair; one point
     more than the minimum lets the residual verify the fit.
   - **Use as is** keeps the ROIs at their pixel positions.
   Accept when the overlay on the right lines up with the arena.
4. If several videos share a frame size, you can reuse one alignment for all of
   them (same camera), or align each separately.

The same window opens automatically when you start a run and some videos have a
different frame size than the ROIs, and through **Check alignment** while you
review each video. The aligned ROIs are saved with a batch plan, and the
reference line and cm calibration follow the alignment: the real-world length
you entered stays the same, so cm values remain correct. A template whose
reference line was drawn before 3.4 stores only its length, so the length is
scaled and marked *approximate*; redraw the line for exact cm values.

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
with them, and 3.5.0 results are unchanged from 3.4.0. Criterion validation
of the automated scores against blinded manual scoring must be performed per
lab/setup (see section 6 and `VALIDATION_PROTOCOL.md`) before results are
published; until then, review the validation video before interpreting
results.

## 6. Validate the Automated Scores (Validation Menu)

The **Validation** menu lets you check the automated scores against your own
blinded manual scoring, without leaving the application. See
`VALIDATION_PROTOCOL.md` for how to choose the recordings and read the result.

1. **Score a video manually...**: pick the paradigm and the raw video, enter
   your initials and press the hotkeys as events happen (Y-maze `A`/`B`/`C`
   for arm entries, EPM `1`-`4` for the arms and `0` for the center, open field
   `C` to toggle between center and periphery, TST hold `I`/`P`/`A`/`U`). Space
   plays, Left/Right step a frame, Shift+Left/Right jump 10 frames, Ctrl+Z
   undoes. The scoring is blind: no analysis result is shown. Progress is
   autosaved and resumes when you reopen the same video with the same
   initials; **Save annotation files** writes the manual CSVs
   (`*_manual_arm_entries.csv`, `*_manual_epm_totals.csv`,
   `*_manual_zone_intervals.csv` or `*_manual_tst_annotations.csv`) to the
   **Save to** folder, which defaults to the video's folder.
2. **Tracking spot-check...**: choose a run folder, click the animal in random
   frames (50 by default, with a fixed seed that is shown and recorded; the
   tracker's dot is hidden) and read the position error, miss rate and false
   detections. The results are saved as `*_position_spotcheck.csv` and
   `*_position_spotcheck_summary.json`.
3. **Build validation report...**: add the run folder and the folder with the
   manual files, check the video matching, set the acceptance criteria (the
   **Protocol defaults** include ICC >= 0.90) and build. You get one
   self-contained `validation_report.html` and a CSV for each table in the
   output folder. The file formats are described in `OUTPUTS.md`.

## 7. Keyboard Shortcuts and Saved Settings

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
| ROI editor | Click | Add a point to the polygon being drawn (snaps to nearby vertices) |
| ROI editor | Enter, double-click, right-click | Close the polygon; with none open, go to the next zone to do |
| ROI editor | Esc | Cancel the polygon in progress (Esc again closes the editor, asking first) |
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
| Manual scoring | Space | Play or pause |
| Manual scoring | Left, Right | Step one frame |
| Manual scoring | Shift+Left, Shift+Right | Step ten frames |
| Manual scoring | Home, End | First or last frame |
| Manual scoring | Ctrl+Z, Delete | Undo; delete the selected row |
| Manual scoring | A/B/C, 1-4, 0, C, I/P/A/U | Score (depends on the paradigm; shown in the dialog) |
| Spot-check | N | Animal not visible |
| Spot-check | Backspace, Left | Previous frame |

The application remembers folders, window layout, the last paradigm, recent
files and the output folder. They are stored in your user profile with Qt's
QSettings (the registry on Windows), not next to the executable. Set the
`BEHAVIORAL_MAZE_ANALYZER_SETTINGS_FILE` environment variable to the path of
an ini file to store them there instead.

## 8. Developer Checks
If you are running from source, check an installation or a change with:

```bash
python -m unittest discover -s tests
python -m ruff check .
```

Set `QT_QPA_PLATFORM=offscreen` for the tests on a machine without a display.
