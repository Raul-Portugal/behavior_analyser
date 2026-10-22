# Changelog

## 4.0.0

Workspace release. **A new single-window workspace replaces the chain of modal
dialogs. Analysis results are unchanged from 3.5.0**, so 3.5.0 and 4.0.0
results can be pooled. The workspace calls the same batch runner as the
command line and the classic interface: a workspace run produces output files
that are byte-identical to a headless `python -m core.cli run` of the same
videos and settings (sequential and parallel; tested), and the Arena & zones,
Detection and Trials stages store exactly the settings the classic wizard
produces (tested). The classic
interface remains available with `--classic` and **File > Switch to classic
interface**.

> **Scientific caveat:** criterion validation must be performed per lab/setup
> using the Validation tools and `VALIDATION_PROTOCOL.md` before results are
> published. Until you have done it for your recordings, always review the
> validation video and tracking coverage before interpreting results.

### Redesigned workspace

The workspace is a guided flow of six stages, each of which fills the whole
window: **1 Videos, 2 Arena & zones, 3 Detection** (**Motion tuning** for the
Tail Suspension Test), **4 Trials, 5 Review & run, 6 Results**. The ROI editor,
the tuner preview and the trial player are no longer squeezed into a tab: the
main canvas takes at least two thirds of the window and the controls sit in a
narrow column beside it.

- **Guided stages and the stepper.** A stepper at the top shows every stage
  with its state (to do, done, needs attention) and a one-line reason as a
  tooltip; every stage can be opened in any order. A Back / Next footer says
  what is missing ("3 of 12 videos need a start time") and disables Next with
  the reason as a tooltip. The stage you left off at is remembered per project.
  Empty and blocked pages say what to do and link to the stage that fixes it.
- **Live edits, no Apply button.** Whatever you change in Arena & zones,
  Detection or Trials is written to the project at once (slider and vertex
  drags are debounced), autosaved, and flushed - never lost or asked about -
  when you leave the stage, close the project or quit.
- **Per-video trial windows** (**Trials** stage). Every video gets its own start
  and end: a table with status, start, end, duration and source, edited in
  place (`90`, `1:30` or `1:30.5`) or by playing the video and pressing `S` /
  `E` (`Enter` jumps to the next video that still needs a start). A **trial
  protocol** gives the trial length (fixed, or until the video ends) and
  whether trials start with the video or at a moment set per video, and
  **Apply to all videos** makes everything follow it again. For the Tail
  Suspension Test every mouse has its own window. Per-video times are stored
  as layers over the template, so analysis settings and checkpoint hashes are
  identical to a full copy with the same values.
- **Per-video zones** (**Arena & zones** stage). Edit the template's zones for
  all videos, or pick one video and **Adjust zones for this video** (the editor
  opens on that video's own frame), **Auto-align to template** (a rigid
  registration of the video's background that also rescales the calibration) or
  **Reset to template**. Videos whose frame size differs from the one the zones
  were drawn on are flagged "Check alignment", and **Tools > Align template to
  videos...** aligns all of them in one go.
- **Simple detection mode and quick check** (**Detection** stage). Two plain
  controls, **Sensitivity** and **Animal size**, drive the same sliders as the
  collapsed **Advanced** section, so the two always agree and a value you did
  not move is never rewritten. **What the detector sees** switches the preview
  to the segmentation mask, a chip grades the current frame and **Quick check
  all videos** scans every video in the background and gives each a Good /
  Check / Poor chip. The detector's confidence is shown beside the preview
  instead of being drawn into the video frame.
- **Review & run** stage. A readiness checklist (Videos, Arena & zones,
  Detection, Trials) with a status chip, a one-line reason and a **Fix** link
  per row; per-video problems grouped by kind with the stage that fixes them;
  the run options (output folder, parallel videos, time bins in minutes, which
  videos to run, sample sheet, validation videos); and a large **Run analysis**
  button that stays disabled, with a tooltip saying why, until nothing blocks
  the run. While it runs the same stage shows the overall progress and ETA, a
  per-video table, a collapsible log and **Cancel** (after a confirmation);
  when it ends, a summary card offers **View results**, **Open folder** and
  **Resume interrupted run**.
- **Results** stage. The run picker lists date, folder and counts; the batch
  summary, the per-video key numbers (stat tiles), quality warnings, plots in a
  grid and the validation video, the time-bin charts and the run comparison
  follow the theme.
- **Design system with light and dark themes.** One set of colour tokens,
  spacing, type scale, icons and widgets (`gui/workspace/theme.py`,
  `gui/workspace/ui_kit.py`); the workspace follows the operating system's
  light or dark setting live. The classic interface keeps the platform look.
- **Project schema 2 with automatic migration.** `.bmaproj` files now store the
  template plus per-video **layers** (trial window, zone override) instead of a
  full copy of the settings for every video. Schema-1 projects are migrated when
  opened, wherever that is exact; what a layer cannot express stays a full
  copy. Versions before 4.0 cannot open `.bmaproj` files.

### Added

- **Workspace.** One window with the stepper on top, the stage in the middle and
  the Back / Next footer at the bottom; a details drawer (Ctrl+I) describes the
  selected video.
  - **Projects** (`.bmaproj`; `core/project.py`, `core/project_io.py`): a
    project holds the videos, the template, the per-video layers, metadata,
    verification state and the run history, with autosave and recovery after a
    crash, a recent-projects list, and import from 3.x templates, batch plans
    and previous run folders.
  - **Video table** (Videos stage) with a paradigm picker, metadata editing,
    filters and search, relinking of moved videos, and drag-and-drop of
    videos, folders, projects and configurations.
  - **Details drawer** for the selected video: details, paradigm, issues, notes
    and actions (verify, apply template, relink, remove).
  - **First-run welcome card.** When the workspace starts with no project to
    reopen but the classic interface remembers recent templates or batch
    plans, a one-time card offers **Create project**, **Open project**,
    **Import your recent 3.x configuration...** (listing up to three recent
    files) and **Use classic interface**. Dismissing it is remembered.
- **Run.**
  - Run scope: all configured videos, selected videos, or videos not yet done.
  - Pre-run checks (unconfigured or missing videos, paradigm and settings
    validation, sample-sheet conflicts and the batch pre-flight) before
    anything starts; what they find stays on the page until it is fixed.
  - Per-video progress, a filterable log and Cancel.
  - Resume of interrupted runs from their checkpoints.
  - Every run is recorded in the project's run history.
- **Results.**
  - A run picker and a summary table of `batch_summary.csv` that can be
    sorted, filtered, copied and exported.
  - Per-video metrics, plots and warnings.
  - A validation video player. When no validation video exists, the original
    video is shown with overlays drawn from the tracking and TST CSVs.
  - Time-bin charts for a video or averaged per group.
  - **Compare runs**, with changed values marked.
  - `Project.clear_sample_sheet()` and a **Clear** button remove a loaded sample
    sheet and the metadata columns.
- **ROI editor** (`gui/roi_editor/`), a zoomable editor used by the Arena & zones stage
  and by the modal flow.
  - Zoom and pan; vertex drag, insert and delete; move, rotate and scale
    whole polygons.
  - Undo and redo.
  - Zones can be drawn in any order.
  - Snapping to the vertices of neighbouring zones so that zones share
    borders (hold Alt to
    turn it off).
  - Choosing **Redraw** while verifying a video now opens the editor
    preloaded with the existing regions.

### Changed

- Paradigm-specific setup knowledge moved into the Maze hooks and a Qt-free
  `SetupState` (`core/setup_state.py`), so the dialogs and the workspace share
  one implementation.
- The detection tuner, the TST tuner and the setup dialogs are now embeddable
  widgets (`gui/tuners/`, `gui/setup_panels/`); the dialogs are thin wrappers
  around them.
- `gui_main.py` opens the workspace by default. `--classic` opens the classic
  wizard window, and `--startup-check` checks both interfaces.
- `RoiSelectorDialog` is now a compatibility subclass of the new
  `RoiEditorDialog`.

### Fixed

- The overlap check between zones of one category (a warning in the ROI
  validation) failed silently with OpenCV 5, whose point-in-polygon test rejects
  numpy integers; it hands OpenCV plain floats now.
- Intermittent crashes from `QObject.sender()` in the tuner and settings
  dialogs (the fix also shipped in 3.5.0).
- Test-only race fixes (a thumbnail-cache test and a Results clipboard
  test); no change to the application.

### Upgrade notes

- **Bringing 3.x configurations into a project.** Use the welcome card, or
  **File > Import** (**Template...**, **Batch plan...**, **Previous run
  folder...**), or open a template or batch plan with **File > Open project...**
  or by dropping it on the window. Each import creates a new `.bmaproj`
  project; the original files are not modified.
- `.bmaproj` files are new and use schema 2 (schema-1 files from pre-release
  builds are migrated when opened). Versions before 4.0 cannot open
  them. **File > Export batch plan...** writes the configured videos as a 3.x
  batch plan, which gives 3.x compatibility.
- Output files, `batch_summary.csv` and the manifest are unchanged from 3.5.0;
  the project file is a separate file and is never needed to read results.

## 3.5.0

Criterion validation release. **This release adds tools for criterion
validation of the automated scores against blinded manual scoring. Analysis
results are unchanged from 3.4.0**, so 3.4.0 and 3.5.0 results can be pooled.
The validation tools are additive: they read analysis outputs and manual
annotation files and never change an analysis output.

> **Scientific caveat:** criterion validation must be performed per lab/setup
> using the Validation tools and `VALIDATION_PROTOCOL.md` before results are
> published. Until you have done it for your recordings, always review the
> validation video and tracking coverage before interpreting results.

### Added

- **Agreement statistics** (`core/agreement_stats.py`, standard library and
  NumPy only):
  - Cohen's kappa with a confidence interval and the confusion matrix.
  - ICC(2,1) and ICC(3,1) with F-based confidence intervals.
  - Bland-Altman bias and limits of agreement with confidence intervals, plus
    the proportional-bias slope.
  - Pearson and Spearman correlation with confidence intervals and p-values.
  - RMSE, MAE and mean absolute percentage difference.
  - Published reference values are reproduced in the tests: Shrout & Fleiss
    (1979) ICC(2,1) 0.29 and ICC(3,1) 0.71, and Bland & Altman (1986) PEFR
    bias -2.1 with limits of agreement -78 to 74.
- **Comparison tools** (`core/positional_validation.py`):
  - Loading of arm-entry annotations, accepting flexible arm names (`A`,
    `arm_a`, `open`, `closed_arm_2`, ...).
  - Event matching with a time tolerance, reporting precision, recall and F1.
  - Sequence edit distance between entry sequences.
  - Y-maze and elevated plus maze comparisons.
  - Frame-label kappa for zone and state labels.
  - Position errors in px and cm.
  - Loading of the exported `*_tracking_data.csv`.
- **TST validation.** Loading of the exported `*_tst_motion_evidence.csv`
  (`core/tst_validation.py`) and immobility-time agreement between the
  automated evidence and manual frame annotations or stopwatch totals.
- **Validation menu** in the main window:
  - **Score a video manually...** is a blind manual scoring dialog with
    per-paradigm hotkeys (Y-maze, EPM, open field, TST), held-key TST bouts,
    undo, autosave and resume, and the scorer's initials recorded. It writes
    `*_manual_arm_entries.csv`, `*_manual_epm_totals.csv`,
    `*_manual_zone_intervals.csv`, `*_manual_tst_annotations.csv` and a
    scoring sidecar JSON.
  - **Tracking spot-check...** shows randomly sampled frames without the
    tracker's position, asks you to click on the animal and reports the
    position error. The random seed and the sampled frames are recorded so the
    sample can be reproduced.
  - **Build validation report...** writes a self-contained HTML report with
    pooled agreement tables, Bland-Altman and scatter plots, confusion
    matrices, acceptance criteria that you can pre-register, marked PASS or
    FAIL, and the SHA-256 of every input file. Each table is also written as a
    CSV.
  - A reusable frame-accurate OpenCV video player widget
    (`gui/video_player.py`) is used by the scoring and spot-check dialogs.
- **`VALIDATION_PROTOCOL.md`** describes how to select a validation set, do
  blinded scoring with two scorers, compute and interpret the statistics, set
  acceptance criteria, and includes a methods-section template.

### Changed

- The Y-maze alternation and same-arm-return arithmetic moved into one shared
  function (`sequence_alternation_metrics`), so automated and manual entry
  sequences are scored by exactly the same rule. The analysis output is
  identical to 3.4.0.

### Fixed

- The local plot-speed test limit was loosened so it no longer fails on loaded
  machines (test-only; no change to the application).

## 3.4.0

Throughput, open-field and metadata release. **Results for existing analyses
are unchanged compared with 3.3.0**, so 3.3.0 and 3.4.0 results can be pooled.
The new metrics and files are additive: they appear only when the new features
(open-field arena, sample sheet, time bins, alignment) are used. Parallel and
sequential runs produce byte-identical outputs.

> **Scientific caveat:** criterion validation of the automated scores against
> blinded manual scoring is still pending. Always review the validation video
> and tracking coverage before interpreting results.

### Added

- **Parallel batch processing.**
  - A **Parallel videos** control on the main window analyses several videos
    at the same time. It defaults to **Auto** (`min(CPU cores / 4, videos,
    RAM in GB / 3)`, at least 1) and is remembered between sessions; 1 runs
    the videos one after another. A 3-video batch is about 2.2x faster.
  - Videos run in spawned child processes. The batch summary and manifest are
    written in the original video order, and the outputs are byte-identical
    to a sequential run.
- **Headless command line.** `python -m core.cli run --plan PLAN --out DIR
  [--workers N] [--sample-sheet SHEET.csv] [--bin-seconds N]` runs a saved
  batch plan (or a template, with `--videos`) without Qt, through the same
  batch runner as the GUI, so the run folder, batch summary, manifest and
  `run_plan.json` are identical. `--paradigm`, `--videos` and `--quiet` are
  also available. Exit codes: `0` success, `1` at least one video failed,
  `2` bad arguments or plan, `130` cancelled.
  - The packaged executable has a hidden `--selftest-parallel` flag that
    analyses two tiny synthetic videos on two spawned worker processes and
    exits 0 on success, to prove that the process pool starts in a frozen
    build.
- **Open-field arena** (Freestyle / Open Field). Enable **Open field arena**
  in the Freestyle setup and draw the arena boundary first.
  - The center is defined either by a **distance from the wall in cm**
    (default 10 cm, needs a reference line) or by a **share of the arena
    area** (default 25%, also used when the video is not calibrated).
  - New metrics: center time, center entries and latency to center;
    thigmotaxis as a percentage of time and of distance; mean distance to the
    wall; mean speed in the center versus the periphery.
  - New per-frame `*_arena_zones.csv` (debounced region and signed wall
    distance).
  - When an arena is drawn, detection is restricted to it, and the threshold
    percentile is computed over arena pixels only, so an empty floor outside
    the arena does not dilute it. Runs without an arena are unchanged.
  - New optional detection setting `min_absolute_contrast` (grey levels,
    default 0 = off): a floor under the percentile threshold. It is not
    written to saved configurations while it is off.
- **Sample sheet.** A CSV of animal metadata (first column `video`, then
  `animal_id`, `group`, ... as you like), chosen with **Sample sheet...** or
  `--sample-sheet`. Its columns appear in `batch_summary.csv`, in each
  `*_summary.json` (`animal_metadata`) and in the manifest. Matching is never
  guessed: a video with no row, or an ambiguous filename, is reported and the
  run asks before starting (the CLI refuses). The sheet is a run option and
  never changes checkpoint settings hashes.
- **Time bins.** **Time bins** in the main window or `--bin-seconds N` also
  report the metrics per bin: `data/<video>_time_bins.csv` per video and
  `batch_time_bins.csv` per run, in long format. Every paradigm is covered,
  and each metric summed over the bins equals the whole-video total.
  Y-maze alternation counts a triplet of arm entries in the bin of its third
  entry. Off by default, in which case nothing changes.
- **ROI template alignment.** Reuse a template on videos with a moved camera
  or another frame size.
  - **Tools > Align template to videos...** compares each video's background
    with the template's and aligns automatically (ORB features plus RANSAC on
    reference frames) or from landmarks you click; **Use as is** keeps the
    ROIs where they are. The same window opens when a run starts and some
    videos do not match the ROIs' frame size, and through **Check alignment**
    in the per-video verification.
  - Calibration follows the alignment. The reference line's endpoints are now
    stored with the ROIs (`reference_line`, only when known) so the cm scale
    can be recomputed exactly; a template that holds only a length is scaled
    and marked approximate.
  - A camera that has shifted, even at the same frame size, is flagged.
  - Aligned settings are applied per video and saved with batch plans.

### Changed

- **Manifest schema 3.** Each `videos[]` entry now has `settings_hash`,
  `video_fingerprint`, `metadata`, `summary_json` and `status`, and failed
  videos are listed too. The manifest also records the sample sheet path and
  the time-bin length. Schema 3 only adds keys to schema 2.
- The checkpoint settings-hash and video-fingerprint helpers are public
  (`get_settings_hash`, `get_video_fingerprint`); the private names remain as
  aliases. The checkpoint format is unchanged.
- The per-video pipeline is now Qt-free (`core/pipeline.py`, with batch
  orchestration in `core/batch_runner.py`); the GUI worker is only the Qt
  signal adapter. This is what makes the CLI and parallel runs possible.
- The dimension-mismatch pre-flight message now points to
  **Tools > Align template to videos...** (or **Check alignment** while
  verifying) as well as redrawing the ROIs. Pre-flight also warns when a
  calibration length was only scaled by an alignment.

### Fixed

- The open-field center outline now appears in validation videos made by CLI
  and parallel runs, not only by GUI runs on the main process.
- The force-quit prompt can no longer run against a deleted window, which
  aborted the test process on CI. This fix was also released in 3.3.0.

## 3.3.0

Performance and usability release. **Analysis results are identical to
3.2.0.** Every optimization was verified to be exact against a copy of the
3.2.0 algorithms, so 3.2.0 and 3.3.0 results can be pooled in one statistical
comparison.

> **Scientific caveat:** criterion validation of the automated scores against
> blinded manual scoring is still pending. Always review the validation video
> and tracking coverage before interpreting results.

### Performance

- **TST frame pipeline.** Resize, grayscale, stabilization warp, optical flow
  and the history-independent segmentation of upcoming frames run on a thread
  pool, while the temporal state is still applied strictly in order. On a
  1080p benchmark this is about 4x faster with 3 mice (34 to 8.6 ms/frame)
  and about 2.6x faster with 1 mouse.
- **TST memory.** Mask contours are stored only for the frames the validation
  video draws, and only when a validation video was requested.
- **Faster detection.** An exact histogram percentile and a lookup-table
  weighting replace the slower equivalents.
- **Reference frame.** About 1.6x faster with lower peak memory.
- **Smaller read buffer.** The video reader queue holds 16 frames, which saves
  up to about 380 MB.
- **Validation videos.** Frames are read about 2x faster, the TST overlay blend
  is about 20x faster, and overlays are frame-accurate because the seek
  position is now verified.
- **TST plots.** The ethogram and trajectory plots are 2-3x faster.

### Usability

- **Remembered settings.** Folders, window layout, last paradigm, recent files
  and the output folder are remembered between sessions.
- **Resume after a restart.** Every run folder now contains `run_plan.json`.
  The **Resume interrupted run** banner and menu action find a checkpoint
  after a restart, reload the matching plan and continue.
- **Load settings from previous run...** reloads the exact per-video settings
  of an earlier run folder.
- **No accidental loss of work.** Cancelling the ROI selector, detection tuner,
  TST tuner, Freestyle configuration or settings review asks before
  discarding edits. Clear List asks first and keeps the loaded configuration
  unless told otherwise.
- **Resume setup.** After a cancelled wizard, the ROIs, rules and tuning
  collected so far are kept and offered again on the next Start.
- **Cancelling per-video verification** after accepting some videos offers to
  run the verified videos, save a partial plan, or discard.
- **Open results from the app.** **Open output folder** and **Open batch
  summary** buttons on the completion box, an Open button next to the output
  folder, and File menu entries for the output and last run folders.
- **Video list.** Drag and drop files and folders, recursive folder add (result
  folders are skipped), remove and reorder, and per-video status icons.
- **Earlier validation.**
  - The ROI selector enforces the TST one-polygon-per-mouse rule while
    drawing.
  - Choosing a template or plan immediately lists videos whose frame size does
    not match the ROIs.
  - The Start Analysis question and the pre-flight checks are one combined
    dialog.
- **Clearer wording.** Prompts use real buttons (Review each video, Run all
  now, Reuse previous configuration, Resume setup). Tuner sliders show real
  values and units.
- **Keyboard shortcuts.**
  - Main window: Ctrl+O add videos, Ctrl+Shift+O add folder, Ctrl+S save
    template, Ctrl+L load configuration or plan, F5 configure and start
    analysis, Esc cancel analysis (with confirmation).
  - Video list: Del or Backspace remove selected, Alt+Up and Alt+Down reorder.
  - ROI selector: Enter for Next, Backspace or Ctrl+Z to undo.
  - TST tuner: Space play/pause, Left and Right step one frame, Ctrl+Left and
    Ctrl+Right step ten frames.
- **Batch ETA.** The estimated time remaining is shown prominently, and setup
  shows a busy bar instead of made-up percentages.
- **Log.** A log level filter and **Save log...**.

### Added

- EPM now writes `<base>_arm_entries.csv` (`entry_number`, `arm`,
  `timestamp_s`), like the Y-maze.
- TST motion evidence rows record `source_frame`, the absolute video frame.
  It is kept in memory only; it is not written to the exported files (see
  `OUTPUTS.md`).

### Fixed

- TST validation now aligns annotations with the correct frames when the
  analysis starts after frame 0.
- `build_release.ps1` now fails on errors instead of reporting success, and it
  refuses to overwrite a running executable.

### Engineering

- A tag-triggered GitHub Actions release build attaches the executable and its
  SHA-256 checksum to the GitHub Release.
- The TST test modules run faster without relaxing any assertion.

## 3.2.0

Scientific correctness, robustness and workflow release. Several outputs
change numerically relative to 3.1.x; read the first subsection before
comparing new results with earlier ones.

> **Scientific caveat:** criterion validation of the automated scores against
> blinded manual scoring is still pending. Always review the validation video
> and tracking coverage before interpreting results.

### Results that differ from 3.1.x

Re-analysing the same video with 3.2.0 can give different numbers. Do not
pool 3.1.x and 3.2.0 results in one statistical comparison without
re-analysing the older videos.

- **Detection tuner resolution.** The detection tuner now matches the
  production 640-px analysis resolution. Detector settings tuned in an earlier
  release may behave differently; review them in the tuner again.
- **Sub-pixel centroids.** Tracked positions are kept as sub-pixel floats
  end to end instead of being rounded to whole pixels. Total distance and
  speed change slightly.
- **Excluded-jump distance is in cm when calibrated.** The distance of
  excluded implausible jumps was previously exported in pixels while labelled
  with the calibrated unit; it is now converted with the other distances.
- **Freestyle immobility.** Immobility now uses smoothed speed, a cm/s
  threshold and a minimum bout length. It is reported as N/A when the video is
  not calibrated. It previously compared raw px/s against the threshold and
  was effectively about 0%.
- **No ROI bounding-box calibration fallback.** The scale is no longer guessed
  from the ROI bounding box. Old configurations that relied on it are flagged
  in the settings dialog, and Redraw ROIs recomputes the scale from the cm
  length you enter. Always draw a reference line to get distances in cm.
- **TST engine v3** (`tst_motion_decomposition_v3`):
  - motion features are normalized to a 30 fps reference rate;
  - the tuner and production analysis share the same 480-px analysis
    geometry;
  - temporal history is reset after unscorable frames.

  TST configurations and tuner profiles saved with engine v2 must be
  re-tuned; checkpoints from v2 runs cannot be resumed. TST analysis is about
  33x faster.
- **Y-maze and EPM** report `entries_possibly_missed`, the number of arm
  entries that tracking gaps may have hidden. Treat arm-entry counts of videos
  with a non-zero value with caution.
- **Batch plans are keyed by relative path.** Saved batch plans store the
  video path relative to the plan file, and a video that does not match a
  plan entry is never given another video's settings. Older filename-keyed
  plans still match when the filename is unambiguous.
- **A read stall now fails or warns.** A decode stall is no longer treated as
  the end of the video: the run fails or warns instead of silently truncating
  the analysis.
- **Checkpoint format v4.** Checkpoints written by earlier releases are
  ignored and the analysis starts from the beginning.
- **Error dialog instead of a crash.** Unexpected errors are shown in an error
  dialog (and logged) instead of closing the application.

### Fixed

- Shared analysis geometry between the analyzer and the tuner, so tuned
  settings match what production runs.
- Decode stalls are no longer mistaken for the end of a video.
- Per-video output-directory failures are recorded and no longer abort the
  rest of the batch.
- The GUI and the worker no longer overwrite each other's reference-frame
  cache.
- Tuner first render, resource cleanup and Quick Scan freezes.
- Settings dialog time sliders, validation and TST detection.
- TST tuner crashes; the state display mapping is shared with production.
- TST reader is released after a failed first read, with a warning when the
  video ends early.
- Closing the window during a run no longer blocks the GUI; every long phase
  of a run checks for cancellation.
- Video captures held by dialogs are released on accept, reject and close,
  and modal dialogs are freed after use.
- The setup-wizard reference frame is generated off the UI thread.
- Y-maze/EPM arm entries hidden by tracking gaps are flagged instead of
  silently dropped.
- Plan-by-filename collisions when two videos share a name in different
  folders.
- Import-time global side effects removed; lint findings and dead code
  cleaned up.

### Changed

- `*_speed_data.csv` now holds raw and smoothed speed and an `is_immobile`
  column.
- Freestyle batch summary adds `immobility_bouts` and
  `smoothed_total_distance`.
- Tracking CSV x/y coordinates are written with 2 decimals.
- TST summaries report the window actually covered versus requested,
  `window_truncated` and `bouts_split_by_uncertainty`.
- Checkpoints are append-only (a small `*_checkpoint.json` header plus a
  `*.rows.jsonl` row log) and saved on a wall-clock interval, so saving cost
  no longer grows with run length.
- Preflight runs once per batch with isolated checks and summed disk need.
- Paradigms can opt out of the reference-line prompt (TST skips it, all
  others offer it).
- One shared Qt image conversion helper (`gui/qt_image.py`) replaces six
  copies of the BGR-to-pixmap code.
- `requirements.txt` has lower bounds matching `requirements-lock.txt`; the
  stale tkinter note was removed.

### Added

- Freestyle dialog: speed smoothing window, immobility threshold (cm/s),
  minimum immobility bout and minimum zone dwell, radio-style mode selection
  and protection of reserved zone names.
- Y-maze/EPM entry options dialog: minimum dwell time and whether the
  starting arm counts as an entry.
- Smoothed-speed, immobility-bout and smoothed path-length kinematics.
- Continuous-integration test workflow and `ruff` configuration
  (`ruff check .`).
- Global exception handler with a user-facing error dialog.

## 3.1.2

See the repository history before the 3.2.0 branch.
