# Changelog

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
