# Changelog

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
- **Excluded-jump distance is in cm when calibrated.** The implausible-jump
  limit is scaled with the other distances, so a calibrated run now applies
  and reports it in cm, not pixels.
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
