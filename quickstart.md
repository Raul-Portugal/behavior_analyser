# Quickstart Guide

This guide walks you through a complete analysis in the 4.0 workspace, from
creating a project and adding your first videos to reviewing the heatmaps and
data. If you prefer the 3.x wizard window, see the
[Classic interface](#appendix-classic-interface) appendix at the end.

**What's new in 4.0:** one window instead of a chain of dialogs. A `.bmaproj`
**project** keeps your videos, their settings and the history of your runs; the
**Setup**, **Run** and **Results** tabs take you from drawing regions to
reading results. Analysis results are unchanged from 3.5.0, and a workspace
run writes the same files as the command line. See `CHANGELOG.md`.

**Earlier releases:** 3.5 added the **Validation** menu (section 8), 3.4 added
parallel videos, the command line, the open-field arena, sample sheets, time
bins and ROI template alignment, and 3.3 added faster analysis, remembered
settings, resume after a restart and keyboard shortcuts. Results are unchanged
since 3.2.0 apart from the additive features; see `CHANGELOG.md`.

## 1. Launch the Application
- **If using the executable (.exe):** Double-click the downloaded file.
- **If using Python:** Open your terminal, activate your virtual environment,
  and run `python gui_main.py`. Add a project file to open it straight away
  (`python gui_main.py study.bmaproj`), or `--classic` for the 3.x wizard window.

The window has the **video table** on the left, the **Setup / Run / Results**
tabs in the middle and a **context panel** on the right that describes the
selected video (resolution, settings, issues, notes). **Help > Quickstart
guide** opens this guide.

## 2. Create or Open a Project
- **File > New project...** (Ctrl+N) asks for a name and a place for the
  `.bmaproj` file and creates an empty project.
- **File > Open project...** (Ctrl+O) opens a project; **File > Open recent**
  lists the last ones. When the application starts it offers to reopen the
  project you were working on.
- Changes are saved with **File > Save** (Ctrl+S) or **Save as...**
  (Ctrl+Shift+S). While you work, the project is also autosaved next to the
  file (`<name>.bmaproj.autosave`); if the application ended before you saved,
  opening the project offers to **recover the autosave**.
- **Coming from 3.x?** The first time the workspace starts with no project and
  you have recent 3.x templates or batch plans, a welcome card offers
  **Create project**, **Open project**, **Import your recent 3.x
  configuration...** and **Use classic interface**. You can also use
  **File > Import > Template...**, **Batch plan...** or **Previous run
  folder...** at any time (or drop a `.json` file on the window). Each import
  creates a new project and leaves the original file untouched.

## 3. Add Videos
1. Use **Add videos...** (Ctrl+Shift+V) or **Add folder...** (Ctrl+Shift+O) in
   the toolbar, or simply drag videos and folders onto the window. Tick **Include subfolders** to take the videos inside the folders
   of a folder as well.
2. Choose the **Paradigm** (Y-Maze, Elevated Plus Maze, Tail Suspension Test or
   Freestyle / Open Field) in the context panel. Changing it later clears the
   settings of configured videos, after asking.
3. The table shows each video's status (New, Configured, Done, Failed, Warning,
   Interrupted). Search and filter by status, double-click a metadata cell to
   edit it, and use **Add column** for your own metadata (group, dose, ...). A
   video that moved shows an issue in the context panel: **Relink...** points
   the project at its new folder. Del removes the selected videos, Alt+Up and
   Alt+Down reorder them.

*All output (CSVs, plots, timelapses) goes in an `analysis_run_...` folder
inside the output folder chosen in the Run tab.*

## 4. Setup
In the **Setup** tab the configuration steps of the chosen paradigm are listed
on the left: parameters, regions, calibration, detection or TST tuning, time
window and review. A green tick means done, a grey ring not done yet and a red
mark a problem, which is listed under the step. Steps can be opened in any
order; **Previous step** and **Next step** walk the list.

- **What you are editing** is shown at the top (**Edit:**). *Project template*
  is the shared setting every video follows; changing it updates all the videos
  that use it. Switch the scope to *Selected video(s)* to give the videos
  selected in the table their own settings, starting from the template's
  values. A video that is edited this way stops following the template until
  you use **Apply template settings** (context panel or right-click menu of the
  table) on it again. With several videos selected, each step you apply is
  copied to all of them, and everything else of each video stays as it was.
- **Apply** writes the open step into the project. Nothing changes until you
  apply; leaving the step, switching scope or selection, closing the project
  or the window with unapplied edits asks whether to apply, discard or stay.
  **Revert step** drops the edits of the open step.
- The **regions** and the tuners work on a frame of the *reference video*:
  the first video of the project (pick another in the *Reference video* box)
  for the template, or the selected video itself.
- **Mark selected videos verified** records that you checked their regions;
  the review step also lists the problems found in the project's videos.

The settings built here are identical to the ones the classic wizard produces
for the same choices.

### Regions of interest (ROI editor)
The **ROI editor** shows a frame from the middle of the reference video. The
**Zones** list on the left has one row per zone (the arena first, if it is
switched on), and a **Reference line** row at the bottom. Each row shows whether
the zone is done (✓), missing (○) or breaks a rule (✗, for example a TST mouse
with two polygons, or a polygon smaller than 100 px²). Click any row at any
time to draw or edit that zone: there is no fixed order.

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
shown on the line. In the **Calibration** step, **Draw reference line...** jumps
to the editor with the line tool selected, and **Real-world length** takes the
length of the line in cm. This allows the software to output distances in
centimeters instead of pixels. **Always draw a reference line when you want
cm**: the software no longer estimates a scale from the ROI bounding box. The
TST does not use calibration.

Apply the step once every zone is done; the editor lists the zones that still
need work.

### Paradigm parameters
- **Y-Maze / Elevated Plus Maze:** set the **Minimum dwell time** (a zone change must last this long to count, which ignores flicker across a zone border) and whether to **count the arm the animal starts in as an entry**.
- **Tail Suspension Test:** the **Number of mice**, the **Minimum immobility bout** and the **Maximum bridged gap**.
- **Freestyle / Open Field:** this step also holds the zones and the immobility parameters: the zone names, the **Immobility threshold** (cm/s), the **Speed smoothing window** (s), the **Minimum immobility bout** (s) and the **Minimum zone dwell** (s). Immobility is scored from the smoothed speed, and is reported as N/A if the video is not calibrated.

- **Open-field arena (Freestyle):** tick **Open field arena (center/periphery)**
  in this step to get center/periphery metrics and thigmotaxis. The ROI editor
  then lists the **arena boundary** first (the floor outline, drawn like any
  zone, before the zones and the reference line). Choose how the center is
  defined: a **distance from the wall in cm** (default 10 cm, needs the
  reference line) or a **share of the arena area** (default 25%; used
  automatically when the video is not calibrated). The ROI editor previews an
  area-share center. The run reports center time, center entries, latency to
  center, thigmotaxis (% of time and of distance), mean distance to the wall
  and center versus periphery speed, and writes `*_arena_zones.csv`. Detection
  is restricted to the arena, and the center outline is drawn in the
  validation video. Zones are optional in this mode; `arena` is a reserved
  zone name. See `OUTPUTS.md`.

### Detection tuning
The Interactive Detection Tuner ensures the software cleanly separates the animal from the background.
1. Use the **Frame Navigation** slider (or the `N` and `P` keys) to look at different parts of the video.
2. Adjust the **Threshold**, minimum component area, and morphology controls
   until the green selected component in the "Threshold View" covers the
   animal without selecting reflections or shadows.
3. Check the evidence-based confidence and quality panel (**Detection Quality
   (Live)**). “Review” or “Poor” means the validation output must be inspected
   and the settings may need adjustment.
4. *(Optional)* Click **Quick Scan Test** to test deterministic sequential
   windows from the beginning, middle, and end of the video.
5. Click **Apply**.

For TST, the **TST motion tuning** step is different: select each mouse in turn
and review the state overlay across the scan. Purple is coherent passive swing,
red is static immobility, green is active escape, and orange/yellow is
uncertain and excluded from scoring. The tuner reviews up to five minutes to
remain responsive; the actual analysis uses the entire selected interval.

### Time window and review
- **Time window:** adjust the **Start** and **End** time (useful for trimming
  setup/handling time); for TST, each mouse can have its own window.
- **Review:** shows what will run and what blocks it. Tick **Create validation
  video (annotated)** (highly recommended for verifying results) to write an
  annotated video for the videos that follow these settings.

### Reusing ROIs when the camera moved or the resolution differs

A saved template or plan holds ROIs in the pixels of the video they were drawn
on, so a video with another frame size, or one recorded with a slightly moved
camera, would get misplaced regions. The pre-run check still stops a run whose
frame size does not match, but you can **align** the template instead of
redrawing it. If a selected video's frame size differs from the one the
template was drawn on, the Setup tab shows a warning with **Align
template...**; the same window opens from **Tools > Align template to
videos...**, which aligns the selected videos (or every video that does not
fit) and stores the result as their own, "aligned" settings.

1. Choose **Tools > Align template to videos...**. The application asks which
   video the template was drawn on (it remembers it for the session), compares
   each video's animal-free background with that video's, and lists the videos
   that need alignment: a different frame size, or a camera that moved so the
   ROIs would be off by more than 3 px. Videos that already fit are left alone.
2. For each flagged video the alignment window shows the template frame with
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
3. If several videos share a frame size, you can reuse one alignment for all of
   them (same camera), or align each separately.

The aligned ROIs are saved with the project (and with an exported batch plan),
and the reference line and cm calibration follow the alignment: the real-world
length you entered stays the same, so cm values remain correct. A template
whose reference line was drawn before 3.4 stores only its length, so the length
is scaled and marked *approximate*; redraw the line for exact cm values.

## 5. Run
The **Run** tab starts the analysis of the project's videos and follows it. It
runs exactly what the classic window and the command line run: the same files
come out for the same settings.

- **Output folder.** Each run creates an `analysis_run_...` folder inside it.
  It defaults to `runs/` next to the project file; **Change...** picks another
  folder and the workspace remembers it for that project.
- **Run.** *All configured videos*, *Selected videos* (the rows selected in the
  table) or *Videos not yet done* (no finished analysis yet). Videos without
  settings are left out of *All* and *Not yet done* with a warning, and block a
  run of *Selected videos*: configure them in the Setup tab first.
- **Parallel videos** sets how many videos are analysed at the same time.
  **Auto** (the default) picks `min(CPU cores / 4, videos, RAM in GB / 3)`, at
  least 1; 1 runs the videos one after another. Outputs are byte-identical to
  a sequential run.
- **Time bins** also reports the metrics in consecutive bins of N seconds (Off
  by default; see `OUTPUTS.md`). Both settings are saved in the project.
- **Sample sheet...** attaches animal information to the results. A sample
  sheet is a CSV whose first column is `video` (a bare filename, or a path
  relative to the sheet) followed by any columns you like:

  ```text
  video,animal_id,group,sex
  mouse1.mp4,A01,control,F
  mouse2.mp4,A02,treated,M
  ```

  The metadata line shows the sheet and how many videos have values; the
  values are loaded into the video table. Those columns appear in
  `batch_summary.csv` (after the shared columns), in each video's
  `*_summary.json` and in `manifest.json`. Videos with no row, or whose
  filename matches several rows, are listed, and you are asked before starting
  with blank metadata for them.
- **Validation videos** has three states. Partly filled (the default) keeps each
  video's own setting from Setup; checked or unchecked creates or skips the
  validation video for every video of *this run* only. The project's settings
  are not changed.
- **Run analysis** (also F5) first checks the plan in one go: videos without
  settings or whose file is missing, the paradigm, the paradigm's own checks,
  metadata columns that clash with output columns, and the usual pre-flight
  checks (video, disk and system). Problems stop the run and are listed;
  warnings are listed once and you confirm them.
- While it runs, the table shows each video's status, progress, ETA and last
  message (a percentage per video is only known when videos run one after
  another), with an overall progress bar and ETA and a log you can filter by
  level and text (**Save log...** keeps it). The video table is locked.
  **Cancel** asks first and saves a checkpoint. Closing the window during a run
  asks, cancels, and closes by itself once the run has stopped.
- When it ends, the run is added to the project's run history (the table
  statuses update) and a summary offers **Open results** (the Results tab,
  showing that run) and **Open folder**.
- **Resume interrupted run** appears when the output folder holds checkpoints
  of project videos whose settings are unchanged; it continues them from where
  they stopped, even after closing the application.

### Running from the command line
A saved batch plan (**File > Export batch plan...** in the workspace, or the
`run_plan.json` of an earlier run) can be analysed without the GUI, for example
on a server:

```bash
python -m core.cli run --plan plan.json --out results --workers 2
```

The run folder is created inside `results` and is identical to a GUI run.
Options: `--workers N` (default Auto), `--sample-sheet`, `--bin-seconds`,
`--quiet`, and `--videos` (plus `--paradigm`) when you pass a template instead of
a plan. Exit codes: `0` success, `1` a video failed, `2` bad arguments or
plan, `130` cancelled. Headless with metadata and bins: `python -m core.cli run
--plan plan.json --out DIR --sample-sheet sheet.csv --bin-seconds 60`.

## 6. Results
The **Results** tab opens a finished run and shows what it produced. It lists
the project's runs, newest first; **Open run folder...** opens any other run
(manifest schema 3, and older runs as far as they were recorded), and **Show
folder** reveals it on disk.

- **Videos.** `batch_summary.csv` as a table: click a header to sort (numbers
  sort as numbers), type in **Filter the table...** to filter every column,
  select cells and press Ctrl+C (or **Copy**) to copy them, or **Export
  CSV...** to save the rows as shown. A warning badge marks videos with quality
  warnings.
- **The selected video** has an *Overview* (key metrics from its
  `summary.json`, the sample-sheet values and the quality warnings), *Plots*
  (thumbnails; click one to enlarge it) and *Video*. The video tab plays the
  validation video when the run made one. Otherwise it plays the original video
  with the analysis drawn on it: the tracked position, its trail, the zone and
  the ROIs for tracking paradigms, and each mouse's state in the usual colours
  for the Tail Suspension Test.
- **Time bins.** When the run had time bins, chart one metric over the bins per
  video, or choose **Group by** a metadata column to average each group (error
  bars are the standard error of the mean; the legend gives n).
- **Compare runs.** Choose two runs (for example before and after re-analysing)
  and compare their batch summaries video by video. Changed values are
  highlighted and show both values; **Only changed videos** hides the rest.

### The run folder
Everything is also on disk. Open the new `analysis_run_...` folder inside your
output folder. It contains `batch_summary.csv`, a `manifest.json` run index,
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
with them, 3.5.0 results are unchanged from 3.4.0, and 4.0.0 results are
unchanged from 3.5.0. Criterion validation of the automated scores against
blinded manual scoring must be performed per lab/setup (see section 8 and
`VALIDATION_PROTOCOL.md`) before results are published; until then, review the
validation video before interpreting results.

## 7. Sharing Work with 3.x
**File > Export batch plan...** writes the configured videos of the project as
a 3.x batch plan, which the classic interface and the command line can load.
Videos without settings are left out (you are told how many). `.bmaproj` files
use schema 1 and cannot be opened by versions before 4.0.

## 8. Validate the Automated Scores (Validation Menu)

The **Validation** menu (in both interfaces) lets you check the automated
scores against your own blinded manual scoring, without leaving the
application. See `VALIDATION_PROTOCOL.md` for how to choose the recordings and
read the result.

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

## 9. Keyboard Shortcuts and Saved Settings

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
| Manual scoring | Space | Play or pause |
| Manual scoring | Left, Right | Step one frame |
| Manual scoring | Shift+Left, Shift+Right | Step ten frames |
| Manual scoring | Home, End | First or last frame |
| Manual scoring | Ctrl+Z, Delete | Undo; delete the selected row |
| Manual scoring | A/B/C, 1-4, 0, C, I/P/A/U | Score (depends on the paradigm; shown in the dialog) |
| Spot-check | N | Animal not visible |
| Spot-check | Backspace, Left | Previous frame |

The application remembers folders, window layout, the last paradigm, recent
files, recent projects, the project that was open last, each project's run
output folder and whether the welcome card was dismissed. They are stored in
your user profile with Qt's QSettings (the registry on Windows), not next to
the executable. Set the `BEHAVIORAL_MAZE_ANALYZER_SETTINGS_FILE` environment
variable to the path of an ini file to store them there instead.

## 10. Developer Checks
If you are running from source, check an installation or a change with:

```bash
python -m unittest discover -s tests
python -m ruff check .
```

Set `QT_QPA_PLATFORM=offscreen` for the tests on a machine without a display.

## Appendix: Classic interface

The 3.x wizard window runs the same analysis and is still available: start it
with `python gui_main.py --classic`, or choose **File > Switch to classic
interface** in the workspace. Here the settings are not kept in a project: you
add videos, run a chain of setup dialogs and start the batch.

### Load your videos
Click **Add Videos** or **Add Folder** at the top of the main window (Ctrl+O and
Ctrl+Shift+O). *All output files (CSVs, plots, timelapses) are saved in the
same directory as the first video in your list, unless you change it via
**Tools > Set Output Directory...**.*

### Configure
Click **Configure & Start Analysis** (F5, or **Run > Configure and Start
Analysis**). The same building blocks as in the Setup tab appear as dialogs, one
after the other:

1. **Select the maze type** (Y-Maze, Elevated Plus Maze, Tail Suspension Test
   or Freestyle / Open Field).
2. **Draw and edit the regions of interest** in the ROI editor, with the
   reference line for calibration. Pressing **OK** without a line asks once
   whether to draw one. **Cancel** asks before it throws away changes. See
   "Regions of interest" in section 4.
3. **Paradigm options** (dwell time, initial arm, immobility parameters,
   open-field arena), see "Paradigm parameters" in section 4.
4. **Tune detection** with the Interactive Detection Tuner (or the TST motion
   tuner) and click **Accept Parameters**.
5. **Final review:** adjust the **Start Time** and **End Time**, enter the
   real-world length of the calibration line and tick **Create validation video
   (annotated)**.

### Run the batch analysis
- If you have multiple videos loaded, the software asks whether to **Review
  each video…** or **Run all now**.
  - **Review each video:** adjust start/end times or **Redraw ROIs** if the
    camera moved between recordings: the ROI editor opens with the current
    ROIs, arena and reference line already loaded, so you only move what
    changed.
  - **Run all now:** applies the exact same ROIs and settings to every video in
    the list.
- A single summary box lists the settings and any pre-flight warnings. Click
  **Start analysis** to begin. The progress bar shows the estimated time
  remaining for the batch.
- To stop, click **Cancel Analysis** (or press Esc) and confirm. The software
  saves a checkpoint, and you can resume from the exact same frame later.
- **Parallel videos** (next to the Start button) works as in the Run tab.
  **Sample sheet...** and **Time bins** (in minutes here) are under the Start
  button; they behave as described in section 5.

### Save, load and resume
- **File > Save Template...** (Ctrl+S) and **File > Load Configuration /
  Plan...** (Ctrl+L) save and load the settings of the batch. A configuration
  or plan can also be turned into a project with **File > Import** in the
  workspace.
- **File > Load settings from previous run...** reuses the settings of an
  earlier run for new videos.
- Every run folder contains a `run_plan.json` with the exact settings of the
  run. If a run was interrupted, even by closing the application, reopen the
  application (or add the same videos again): a **Resume interrupted run**
  banner (also **File > Resume interrupted run...**) reloads the matching plan
  and continues from the saved frame.
- To align a template to videos whose camera moved, load the template with
  **File > Load Configuration / Plan...**, add the videos and choose **Tools >
  Align template to videos...** (see "Reusing ROIs when the camera moved" in
  section 4). The same window opens automatically when you start a run and some
  videos have a different frame size than the ROIs, and through **Check
  alignment** while you review each video.

### Classic keyboard shortcuts

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

The editors, tuners and validation dialogs use the shortcuts of section 9.
