# Analysis output contract

Each analysis creates one timestamped run folder:

```text
analysis_run_YYYYMMDD_HHMMSS/
  manifest.json
  run_plan.json
  batch_summary.csv
  video_name/
    data/
    plots/
    media/          # only created when a validation video is requested
```

The manifest identifies the application version, paradigm, source videos,
effective algorithm parameters, failures, quality warnings, and every
generated artifact. This makes each run self-contained and prevents files from
separate runs or videos being mixed.

## Run plan (`run_plan.json`)

Every run folder also contains `run_plan.json`, written when the run starts.
It is a batch plan (the same format as a saved batch plan, see "Batch plans"
below) holding the exact per-video settings of that run: ROIs, detection
settings, time window, scale, paradigm and algorithm parameters. Its keys are
paths relative to the run folder.

The application uses it in two ways:

- **Resume after a restart.** A checkpoint only resumes when its settings hash
  matches the settings of the next run. At startup, and whenever videos are
  added, the application pairs each checkpoint with the run plan whose entry
  for that video hashes to the same value. **Resume interrupted run** then
  reloads that plan and the paradigm parameters, so the checkpoint continues
  exactly where it stopped. Checkpoints that no run plan explains (for
  example from runs started by an older version) are not resumed.
- **File > Load settings from previous run...** loads the plan of an earlier
  run folder so the same settings can be applied to new videos.

`run_plan.json` is a settings file, not a result: it contains no analysis
output. If it cannot be written the analysis still runs (the failure is only
logged), but that run cannot be resumed after a restart.

## Shared batch columns

Every paradigm uses the same leading columns:

- source filename and paradigm
- analyzed duration and frame count
- total distance, excluded jump count/distance, and distance unit
- tracking quality, detection rate, mean confidence, and undetected time

Distance and tracking-quality cells are intentionally blank for TST because
motion-energy analysis does not produce centroid tracking. A blank unsupported
metric must not be interpreted as zero.

## Per-video data

| Artifact | Positional paradigms | TST | Purpose |
|---|---:|---:|---|
| `*_tracking_data.csv` | yes | no | Frame-level positions and raw/cleaned zones |
| `*_tst_motion_evidence.csv` | no | yes | Per-frame active-motion evidence, rigid/angular/residual motion, pendulum evidence, behavioral state, confidence, and QC for each mouse |
| `*_analysis_summary.csv` | yes | yes | Human-readable key results |
| `*_summary.json` | yes | yes | Complete machine-readable result and configuration |
| `*_arm_entries.csv` | Y-maze, EPM | no | Debounced, center-mediated arm-entry events |
| `*_speed_data.csv` | Freestyle | no | Per-frame raw and smoothed speed and the `is_immobile` flag (see below) |

Raw and cleaned zone labels are both retained in positional tracking CSVs.
Behavioral metrics use the cleaned labels; the raw labels remain available for
quality review.

## Plots by paradigm

| Paradigm | Generic plots | Specific plots |
|---|---|---|
| Y-maze | heatmap, trajectory, zone time series | arm sequence, transition matrix |
| Elevated Plus Maze | heatmap, trajectory, zone time series | none; the shared plots already cover its spatial outputs |
| Freestyle / Open Field | heatmap, trajectory, zone time series | speed; transition matrix and entry counts when zones exist |
| Tail Suspension Test | none | active-evidence ethogram with immobile and uncertain periods |

EPM arm distances do not necessarily sum to total distance: movement in the
center or outside all ROIs is reported separately. The batch `total_distance`
column is the complete tracked path and is therefore the appropriate value for
cross-video comparisons.

Implausible jumps above the configured frame-diagonal limit are excluded from
all path-distance metrics. Their count and excluded raw distance are reported
explicitly rather than silently inflating totals. The excluded distance is in
the run's distance unit: cm when a reference line was drawn, pixels otherwise.

Validation media is optional and lives in `media/`; it is not created merely
because plots or tabular outputs are generated.

For TST, the existing summaries include quality-control fields for body-mask
stability, segmentation failures, low-confidence frames, and state switching.
They are deliberately not written as an additional per-video QC file.

## Column reference (3.2)

### Positional tracking CSV

`*_tracking_data.csv` writes the x and y centroid with 2 decimals. Positions
are sub-pixel values, so rounding them to whole pixels is not needed and not
done.

### Freestyle / Open Field

`*_speed_data.csv` columns:

| Column | Meaning |
|---|---|
| `frame`, `timestamp_s` | Frame index and time |
| `speed_<unit>_per_s` | Raw frame-to-frame speed (`<unit>` is `cm` when calibrated, otherwise `px`) |
| `smoothed_speed_<unit>_per_s` | Speed over the centered smoothing window; blank when no valid value exists |
| `is_immobile` | `1` when the frame is inside an immobility bout, `0` otherwise; blank when immobility is unavailable |
| `distance_unit` | `cm` or `px` |

Immobility is scored from the smoothed speed against a cm/s threshold, and
only bouts of at least the configured minimum length count. When the video is
not calibrated there is no cm/s speed, so immobility time, percentage and bout
count are written as `N/A` (and `is_immobile` is blank) instead of a number.

The Freestyle batch summary adds, after the speed and immobility columns,
`immobility_bouts` (number of bouts) and `smoothed_total_distance` (path
length of the smoothed trajectory; `total_distance` remains the unsmoothed
path).

#### Open field arena (center / periphery, thigmotaxis)

When **Open field arena** is enabled in the Freestyle configuration, an arena
polygon (the floor boundary) is drawn first, separately from any user zones.
Nothing changes for runs without an arena: the same columns, values and files
are written as before.

The center is the arena shrunk by a distance from the walls. It is defined
either by a distance in cm (default 10 cm, used when the video is calibrated)
or by a fraction of the arena's area (default 0.25, used when chosen or when
the video is not calibrated; the run notes say which rule applied).
Each frame is labelled `center`, `periphery`, `outside` (beyond the wall) or
`undetected`, then debounced with the same minimum zone dwell as user zones.

The batch summary and analysis summary then end with these columns (the
JSON summary repeats them under `freestyle_metrics.arena`, together with the
center rule actually used and its threshold in px and cm):

| Column | Meaning |
|---|---|
| `center_time_s`, `periphery_time_s` | Time in each region (debounced labels) |
| `center_entries` | Transitions into the center. Starting in the center is not an entry, and a tracking gap is never bridged, so this count is conservative |
| `latency_to_center_s` | Time from the start of the analyzed window to the first center frame; `N/A` if the center was never entered |
| `thigmotaxis_time_pct` | `periphery_time_s / (center_time_s + periphery_time_s)` as a percentage |
| `thigmotaxis_distance_pct` | Periphery path length divided by in-arena (center + periphery) path length, as a percentage. Steps follow the same valid-step and excluded-jump rule as `total_distance`; a step belongs to the region where it ends |
| `mean_wall_distance` | Mean distance to the nearest wall over in-arena frames (cm when calibrated, pixels otherwise) |
| `center_speed_mean`, `periphery_speed_mean` | Mean smoothed speed in each region (cm/s when calibrated, px/s otherwise); `N/A` when the region was never occupied |

`*_arena_zones.csv` (arena runs only) lists, per frame, the debounced
`arena_zone` and the signed wall distance (`wall_distance_<unit>`, positive
inside the arena, negative outside, blank when undetected).

When an arena is drawn, detection is restricted to it: background-difference
pixels outside the arena are ignored, and the detection threshold percentile
is computed from the arena pixels only (so an empty floor outside the arena
does not dilute it). Without an arena the whole frame is used, exactly as
before.

### Y-maze and EPM

`*_arm_entries.csv` (Y-maze and EPM) lists the debounced, center-mediated arm
entries in order. Columns: `entry_number` (1, 2, ...), `arm` and
`timestamp_s` (seconds, 3 decimals). The EPM file has the same layout as the
Y-maze file.

The batch summary and analysis summary include `entries_possibly_missed`: the
number of times the animal reappeared after a tracking gap in a way that may
hide an arm entry (it resumes in a different arm than before the gap, or was
last seen in the center, and no center crossing was confirmed afterwards).
Counting stays conservative and does not guess the route taken during the gap,
so a non-zero value means the entry counts for that video are a lower bound;
inspect the validation video.

### Tail Suspension Test

The TST summary adds, per mouse:

- `bouts_split_by_uncertainty`: immobility bouts that were split by a stretch
  of unscored (uncertain) frames, which would otherwise have been counted as
  one bout;
- `window_start`, `window_end`: the interval actually covered by scored
  frames, next to `requested_window_start`, `requested_window_end`: the
  interval that was asked for;
- `window_truncated`: `1` when the covered window is shorter than the
  requested one by more than a frame plus half a second (for example the video
  ends early), otherwise `0`.

The evidence features written to `*_tst_motion_evidence.csv` and the JSON
results are rate-normalized to a 30 fps reference (see `TST_ENGINE.md`), so
values are comparable across recordings with different frame rates.

In memory, each per-frame evidence row also carries `source_frame` (the
absolute video frame it was computed from, which matters when the analysis
window starts after frame 0) and, only when a validation video was requested
and only on the frames that video draws, `mask_contour` (the body-mask
outline). `source_frame` is used internally to align validation annotations
with the correct frames. Neither key is a column of
`*_tst_motion_evidence.csv`, and neither is part of `*_summary.json`: the
evidence CSV is written from a fixed list of columns, and the JSON does not
contain per-frame evidence. The CSV `frame` column is derived from each row's
timestamp and the frame rate. `mask_contour` is a drawing aid only, is not
part of any exported data, and is dropped from checkpoint files (a resumed
run's validation video simply has no mask outline for frames restored from a
checkpoint).

## Batch plans

A saved batch plan is keyed by each video's path relative to the plan file's
folder, with forward slashes (for example `cohort_a/mouse1.mp4`). A video on
another drive is keyed by its absolute path. Older plans keyed by bare
filename still match when the filename is unambiguous. A video that matches no
entry is never given another video's settings; the application reports it as
unmatched and asks whether to configure it separately, skip it or cancel.

## Checkpoints

Resumable runs write to a `.checkpoints/` folder in the output directory, two
files per video:

- `<video>_<id>_checkpoint.json`: a small header (checkpoint version, settings
  hash, video fingerprint, last processed frame, temporal engine state, scalar
  results and the number of committed rows), rewritten atomically;
- `<video>_<id>_checkpoint.rows.jsonl`: one JSON line per processed frame,
  appended and flushed before the header is rewritten.

The checkpoint format is version 4 (unchanged in 3.3). The run folder's
`run_plan.json` is what lets a checkpoint be resumed after the application was
restarted (see "Run plan" above). Checkpoints from other versions, or whose
settings or video do not match, are ignored and the run starts fresh. Both
files are removed when a video finishes.
