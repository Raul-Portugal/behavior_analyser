# Analysis output contract

Each analysis creates one timestamped run folder:

```text
analysis_run_YYYYMMDD_HHMMSS/
  manifest.json
  run_plan.json
  batch_summary.csv
  batch_time_bins.csv   # only when time bins are on (see "Time bins")
  video_name/
    data/
    plots/
    media/          # only created when a validation video is requested
```

The manifest identifies the application version, paradigm, source videos,
effective algorithm parameters, failures, quality warnings, and every
generated artifact. This makes each run self-contained and prevents files from
separate runs or videos being mixed.

**The outputs are unchanged in 4.0.** Every file described below is written
exactly as in 3.5.0, whichever interface started the run: a run from the 4.0
workspace produces the same files as the command line (byte for byte, which the
tests check) and as the classic interface. The new `.bmaproj` project (see
"Projects (4.0)") is a separate file that records which runs were made; a run
folder never needs it and does not refer to it.

## Manifest (`manifest.json`, schema 3)

`schema_version` is `3` (3.4.0; unchanged in 3.5.0 and 4.0.0). Top-level keys: `application`, `paradigm`,
`started_at`, `finished_at`, `cancelled`, `requested_videos`,
`completed_videos`, `sample_sheet` (path of the sample sheet, or `null`),
`time_bin_seconds` (bin length, or `null` when bins are off), `videos`,
`failures`, `quality_warnings`, `algorithm_parameters` and `generated_files`.

Each `videos[]` entry describes one video, completed **or failed**, in the
order of the batch:

| Key | Meaning |
|---|---|
| `source` | Path of the source video |
| `output_folder` | The video's folder inside the run folder |
| `status` | `completed` or `failed` |
| `settings_hash` | Hash of the exact per-video settings (the value checkpoints are validated against; it is the hash of the matching `run_plan.json` entry) |
| `video_fingerprint` | Hash of the source video's path, size and modification time; `null` if the file could not be read |
| `metadata` | The video's sample-sheet values (`{}` without a sample sheet) |
| `summary_json` | Path of the video's `*_summary.json`, relative to the run folder; `null` when it was not written (a failed video) |

Every video of the run, failed ones included, is in `run_plan.json`, so a
failed video can be re-run from the run plan; its `settings_hash` in the
manifest identifies the settings it was run with. Manifests of schema 2 had
only `source` and `output_folder` per completed video; schema 3 only adds
keys.

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
  for that video hashes to the same value. **Resume interrupted run** (a button
  in the workspace's Review & run stage, and a banner in the classic interface) then
  reloads that plan and the paradigm parameters, so the checkpoint continues
  exactly where it stopped. Checkpoints that no run plan explains (for
  example from runs started by an older version) are not resumed.
- **File > Load settings from previous run...** (classic interface) loads the
  plan of an earlier run folder so the same settings can be applied to new
  videos. In the workspace, **File > Import > Previous run folder...** builds a
  project from it.

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

When a sample sheet is used (see "Sample sheet"), its columns follow these
shared columns and precede the paradigm columns.

## Per-video data

| Artifact | Positional paradigms | TST | Purpose |
|---|---:|---:|---|
| `*_tracking_data.csv` | yes | no | Frame-level positions and raw/cleaned zones |
| `*_tst_motion_evidence.csv` | no | yes | Per-frame active-motion evidence, rigid/angular/residual motion, pendulum evidence, behavioral state, confidence, and QC for each mouse |
| `*_analysis_summary.csv` | yes | yes | Human-readable key results |
| `*_summary.json` | yes | yes | Complete machine-readable result and configuration |
| `*_arm_entries.csv` | Y-maze, EPM | no | Debounced, center-mediated arm-entry events |
| `*_speed_data.csv` | Freestyle | no | Per-frame raw and smoothed speed and the `is_immobile` flag (see below) |
| `*_arena_zones.csv` | Freestyle with an arena | no | Per-frame debounced `arena_zone` and signed wall distance (see "Open field arena") |
| `*_time_bins.csv` | all, when time bins are on | yes | Per-bin metrics in long format (see "Time bins") |

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
because plots or tabular outputs are generated. For an open-field arena run
the tracking validation video also draws the center region outline, whether
the run was started from the GUI, the command line or a parallel batch.

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

The detection settings also have an optional `min_absolute_contrast` (grey
levels, 0-255, default 0 = off): a floor under the percentile threshold, so
that a nearly empty arena does not get a very low threshold. It is written to
saved configurations and to `*_summary.json` only when it is above 0, so
existing configurations and settings hashes are unchanged.

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

## Sample sheet (animal metadata)

An optional CSV that attaches animal information (ID, group, sex, treatment,
...) to each video. The first column is named `video`; every other column is
free metadata, copied as written:

```text
video,animal_id,group,sex
mouse1.mp4,A01,control,F
day2/mouse1.mp4,A01,control,F
```

A `video` cell is a path relative to the sheet's folder, an absolute path, or
a bare filename. Cells are stripped, a UTF-8 BOM is ignored, blank lines are
skipped, and the delimiter may be a comma, a semicolon or a tab. Videos are
matched to rows with the same rules as batch plans, with the sheet file as the
anchor (exact path, then path relative to the sheet, then a filename that is
unique in both the sheet and the selected videos). A video is never matched by
guesswork: one that matches no row, or whose filename fits several rows or is
shared by several selected videos, is reported (the GUI lists them when the
sheet is loaded and asks before starting; the CLI refuses to run). Metadata
column names may not repeat a column of the batch summary or of the
time-bin tables (`video_filename`, `paradigm`, `total_distance`, `metric`,
`value`, ...); such a sheet is rejected.

Where the metadata is written (and nowhere else):

- `batch_summary.csv`: the metadata columns sit **between** the shared columns
  and the paradigm columns. Without a sample sheet the file is unchanged.
- each video's `*_summary.json`: an `animal_metadata` object.
- `manifest.json`: `videos[].metadata` and the top-level `sample_sheet`.
- `batch_time_bins.csv` (when time bins are on): after `video_filename` and
  `paradigm`.

The sample sheet is a run option, not a video setting: it is not part of
`run_plan.json` or of the checkpoint settings hash, so adding or changing it
never invalidates a checkpoint. Select it with **Sample sheet...** in the main
window (the last sheet is remembered) or `--sample-sheet` on the command line.

## Time bins

An optional run setting that reports the metrics in consecutive fixed-length
bins, for example to see habituation over a 10-minute open field. Choose
**Time bins** in the main window (in minutes) or `--bin-seconds N` on the
command line. **Off is the default and changes nothing**: no extra file is
written and every other output stays byte-for-byte what it was. The bin
length is not part of the settings or the checkpoint hash.

When on, each video gets `data/<video>_time_bins.csv` and the run folder gets
`batch_time_bins.csv`. Both are **long format**, one row per bin, subject and
metric (easy to pivot, and the metrics differ per paradigm); the batch file
has the same rows for every video, led by `video_filename`, `paradigm` and the
sample-sheet columns (if any).

| Column | Meaning |
|---|---|
| `bin_index` | 1, 2, 3, ... |
| `bin_start_s`, `bin_end_s` | Bin limits in seconds **from the analysis start** (the first analyzed frame, i.e. the Start Time), not from the start of the video |
| `abs_start_s`, `abs_end_s` | The same limits as absolute video time |
| `bin_duration_s` | Length of the bin; shorter than the bin length only for a partial bin |
| `partial_bin` | `1` for a short last bin, otherwise `0`. The last bin is partial when the analysis ends more than one frame before the bin would; its `bin_end_s` is where the analysis ended |
| `subject` | The mouse for TST; blank otherwise |
| `metric`, `value`, `unit` | The metric (see below), its value (blank when not defined, e.g. a percentage with no data) and its unit |

Every count belongs to exactly one bin, so **a metric summed over the bins
equals the whole-video total** (`value`s are written with 4 decimals, so
totals agree to rounding). A frame (time in a zone, undetected time,
immobility) belongs to the bin of its timestamp; a distance step (from the
previous frame to this one) belongs to the bin and zone of the frame it ends
in, as in the whole-video distance, and excluded jumps are excluded here too;
an entry belongs to the bin of the frame where it happens. Distances are in
cm when calibrated, otherwise pixels.

Metrics by paradigm:

| Paradigm | Metrics |
|---|---|
| All positional paradigms | `distance_total`; `distance_in_<zone>`; `time_in_<zone>_s`; `undetected_time_s` |
| Y-maze | the above, plus `entries_total`, `entries_<arm>`, `alternations_correct`, `alternations_possible`, `alternation_pct` |
| Elevated Plus Maze | the positional metrics, plus `entries_total`, `entries_<arm>`, `entries_open_arms`, `entries_closed_arms` |
| Freestyle / Open Field | the positional metrics, plus `entries_total` and `entries_<zone>` (when zones exist), `immobility_time_s` (calibrated runs only), and with an arena `center_time_s`, `periphery_time_s`, `thigmotaxis_time_pct` |
| Tail Suspension Test | per mouse: `immobile_s`, `mobile_s`, `uncertain_s`, `immobility_pct` (immobile / (immobile + mobile)); no distances |

Notes:

- Arm entries are the same debounced, center-mediated entries as the
  whole-video counts (`analyze_center_mediated_entries`), placed in the bin
  where the entry happens.
- **Y-maze alternation across bins.** A triplet of consecutive arm entries is
  counted in the bin where its **third** entry occurs. A triplet that
  straddles a bin boundary therefore uses entries of the previous bin as
  context but is scored once, in the later bin. `alternations_possible` of a
  bin is the number of triplets ending in it (they sum to the video's possible
  alternations); `alternation_pct` is correct / possible within the bin,
  blank when the bin has no triplet.
- Freestyle zone entries follow the whole-video rule (every transition into a
  zone, plus the zone the animal is first seen in); a zone the animal never
  occupied has no rows. Immobility uses the bout-filtered states of the
  summary, so it needs a calibrated video.
- TST seconds are counted only inside each mouse's scored window; frames
  outside it are in no bin. Bins are still counted from the analysis start,
  so a window that starts later leaves the first bins partly or fully empty.
  `uncertain_s` is time with no scorable state, as in the summary.

## Validation outputs (3.5)

The **Validation** menu writes its own files. They are never part of an
analysis run folder and never change an analysis output. See
`VALIDATION_PROTOCOL.md` for the workflow. `<video>` is the video's file name
without its extension; the files go to the folder chosen in the dialog (by
default the video's folder).

### Manual scoring (Validation > Score a video manually...)

| File | Written for | Content |
| --- | --- | --- |
| `<video>_manual_arm_entries.csv` | Y-maze, EPM | `entry_number,arm,timestamp_s`, the columns of `<base>_arm_entries.csv`. Arms are `arm_a`, `arm_b`, `arm_c` (Y-maze) or `open_arm_1`, `open_arm_2`, `closed_arm_1`, `closed_arm_2` (EPM). Center markers are not entries and are not listed. `timestamp_s` is `frame / fps` |
| `<video>_manual_epm_totals.csv` | EPM | `metric,value` totals derived from the scored arm and center marks (time in open arms, closed arms and center, percent time in open arms) |
| `<video>_manual_zone_intervals.csv` | Open field | `interval_number,zone,start_frame,end_frame,start_s,end_s,duration_s`. `zone` is `center` or `periphery`. The intervals tile the scoring window; frames are inclusive, `start_s = start_frame / fps` and `end_s = (end_frame + 1) / fps` |
| `<video>_manual_tst_annotations.csv` | TST | `mouse_name,frame,state` with one row per labelled frame; states are `immobile_static`, `passive_swing`, `active_escape` and `uncertain` (same format as `TST_VALIDATION.md`) |
| `<video>_manual_scoring_<paradigm>_<initials>.json` | All | The scoring sidecar, autosaved while scoring and used to resume an interrupted session (accepted only for the same paradigm, video and initials). Keys: `schema_version`, `application`, `app_version`, `paradigm`, `video_name`, `fps`, `total_frames`, `scorer_initials`, `date_started`, `date_saved`, `date`, `blind` (always `true`), `roi_file`, `window`, `initial_zone`, `events`, `mice`, `fill_unmarked`, `bouts`, `saved_files`, `csv_up_to_date`. `paradigm` is `y_maze`, `epm`, `open_field` or `tst`; the initials are `ANON` when none were entered |

### Tracking spot-check (Validation > Tracking spot-check...)

| File | Content |
| --- | --- |
| `<base>_position_spotcheck.csv` | One row per reviewed frame: `sample_number, frame, timestamp_s, visible, manual_x, manual_y, auto_x, auto_y, error_px, error_cm, outcome, scale_factor`. Positions are source pixels; `error_cm` is blank when the video is not calibrated. `outcome` is `matched`, `missed` (animal visible, tracker found nothing), `false_detection` (animal not visible, tracker reported a position), `correct_absent` or `unreadable` (frame could not be decoded; not counted) |
| `<base>_position_spotcheck_summary.json` | Provenance (`application`, `app_version`, `date`, `scorer_initials`, `video`, `tracking_csv`, `summary_json`, `blind`) and `results`: `seed`, `requested_samples`, `sampled_frames`, `reviewed`, `unreadable`, `visible`, `not_visible`, `matched`, `missed`, `miss_rate`, `false_detections`, mean, median and 95th percentile error in px and cm, `scale_factor` and `calibrated`. The seed and the sampled frames reproduce the sample |

`<base>` is the `<base>` of the run's `<base>_tracking_data.csv`.

### Validation report (Validation > Build validation report...)

The report folder (default `validation_report_YYYYMMDD_HHMMSS`, next to the
first input) holds:

- `validation_report.html`: one self-contained file (styles and plots
  embedded, nothing loaded from the network) with the per-video tables, pooled
  agreement, event precision and recall, Bland-Altman and scatter plots,
  confusion matrices, the position spot-check, the acceptance criteria and the
  application version, date and input files.
- `validation_report_<table>.csv`: every table of the report, one file each:
  `inputs`, `y_maze_per_video`, `epm_per_video`, `open_field_per_video`,
  `tst_per_mouse`, `pooled_agreement`, `pooled_events`, `frame_agreement`,
  `<name>_confusion` (`open_field_zones_confusion`, `tst_states_confusion`),
  `position_spotcheck_per_video`, `position_spotcheck_pooled`,
  `acceptance_criteria` and `input_files`. Only the tables that have data are
  written.

`acceptance_criteria` lists each criterion (`criterion, key, comparison,
threshold, value, result`) with `result` `pass`, `fail` or `not evaluable`.
`input_files` lists every input (`role, video, path, size_bytes, sha256`), so
the report can be tied to the exact files it was computed from. Differences are
automated minus manual.

## Batch plans

A saved batch plan is keyed by each video's path relative to the plan file's
folder, with forward slashes (for example `cohort_a/mouse1.mp4`). A video on
another drive is keyed by its absolute path. Older plans keyed by bare
filename still match when the filename is unambiguous. A video that matches no
entry is never given another video's settings; the application reports it as
unmatched and asks whether to configure it separately, skip it or cancel.

## Projects (4.0)

A project (`.bmaproj`) keeps everything about one study in a single JSON file:
the videos, the settings of every video, free metadata per video, the run
options and the history of the runs made from it. It is data only: analysis
outputs are never changed by it, and a run started from a project writes the
same run folder, `run_plan.json` and `manifest.json` as any other run (the
outputs are unchanged in 4.0). The run folders are created in the Review & run
stage's output folder, which defaults to `runs/` next to the project file; that folder
is remembered per project by the application's settings, not stored in the
project. The project's `runs` list is the history shown in the Results stage.

```json
{
  "type": "project",
  "schema_version": 1,
  "name": "Cohort A, Y-maze",
  "created_at": "2026-03-02T09:14:00+0000",
  "modified_at": "2026-03-02T11:40:12+0000",
  "app_version": "4.0.0",
  "paradigm": "Y-Maze",
  "algorithm_parameters": {},
  "notes": "",
  "run_options": {
    "parallel_workers": 0,
    "time_bin_seconds": 60.0,
    "sample_sheet": "sheets/cohort_a.csv",
    "create_validation_video": false
  },
  "template_settings": { "schema_version": 5, "roi_manager": {}, "...": "..." },
  "videos": [
    {
      "key": "videos/mouse1.mp4",
      "path": "videos/mouse1.mp4",
      "metadata": { "animal_id": "A01", "group": "control" },
      "settings": "template",
      "settings_source": "template",
      "verified": true,
      "fingerprint": "3f1c0b7a52d94e0c8a1b77e2",
      "last_status": "done",
      "last_run_dir": "analysis_run_20260302_104500",
      "notes": ""
    },
    {
      "key": "videos/mouse2.mp4",
      "path": "videos/mouse2.mp4",
      "metadata": { "animal_id": "A02", "group": "treated" },
      "settings": { "schema_version": 5, "roi_manager": {}, "...": "..." },
      "settings_source": "custom",
      "verified": false,
      "fingerprint": "9d2e41c6b0a85f3d1c7e04aa",
      "last_status": "configured",
      "last_run_dir": null,
      "notes": "camera was bumped"
    }
  ],
  "runs": [
    {
      "run_dir": "analysis_run_20260302_104500",
      "started_at": "2026-03-02T10:45:00+0000",
      "finished_at": "2026-03-02T10:52:31+0000",
      "cancelled": false,
      "completed": 1,
      "failed": 0,
      "manifest": "analysis_run_20260302_104500/manifest.json",
      "settings_hashes": { "videos/mouse1.mp4": "b1a07c2e55d3e9f0a4c18d62" }
    }
  ]
}
```

(`"roi_manager": {}` and `"..."` stand for the usual settings content, see
"Configuration schemas"; they are not abbreviated in a real file.)

- **Paths** (`path`, `last_run_dir`, `run_dir`, `manifest`, `sample_sheet`) are
  stored relative to the project file's folder, with forward slashes, and
  resolved against it on load, so a study folder can be moved or shared as a
  whole. A path with no relative form (another drive) is stored absolute.
- **`key`** identifies a video for good: it starts as the video's path
  relative to the project file (the same key a batch plan uses) and never
  changes when the video is relinked or the project file is saved elsewhere.
  Run history refers to videos by key.
- **`settings`** is a full settings object (the same as a template or one batch
  plan entry), `null` for a video that is not configured yet, or the string
  `"template"` for a video whose settings are exactly `template_settings`
  (written once instead of once per video). `settings_source` is `template`,
  `custom` or `aligned` (the result of aligning the template to this video).
  Changing a video's settings clears `verified` and sets `last_status` back to
  `configured`.
- **`fingerprint`** is the checkpoint video fingerprint (path, size and
  modification time) taken when the video was configured. When it no longer
  matches, the project reports that the video changed since it was configured.
  Relinking a video records it again, since the path is part of it.
- **`last_status`** is `new` (no settings), `configured`, `done`, `warning`
  (done, with a quality warning), `failed` or `interrupted` (listed in the
  run's `run_plan.json` but without an outcome in the manifest). It is set from
  the run's `manifest.json` (schema 3) when the run is recorded.
- **`settings_hashes`** per run are the manifest's per-video `settings_hash`,
  keyed by video key.
- **`run_options.parallel_workers`** `0` means automatic.
  `time_bin_seconds` `0` means no time bins.

The file is written atomically (a temporary file, then a rename). A project
with a newer `schema_version` than the application understands is refused with
a clear message; unknown settings inside a video entry (for example from a
newer settings schema) are kept as they are and written back unchanged.
While a project is open the application writes `<name>.bmaproj.autosave` next
to it a couple of seconds after each change and offers to recover it at the
next start when it is newer than the project file; saving the project removes
it.

Projects convert both ways with the earlier formats. A project can be started
from a template (every video inherits it), from a batch plan (entries are
matched to videos exactly as when loading a plan) or from any 3.x run folder,
using `run_plan.json` for the videos and settings and `manifest.json` for the
metadata, sample sheet, time-bin length, video fingerprints and each video's
outcome. A project can be exported as an ordinary batch plan (**File > Export
batch plan...**, keys relative to the plan file) that 3.x builds and the
command line read; videos without settings are left out of the plan. Versions
before 4.0 cannot open a `.bmaproj` file.

## Reference-line endpoints in configurations

The `roi_manager` block of a saved template, batch plan or `run_plan.json`
keeps the calibration length in `_reference_length_pixels` (source pixels).
From 3.4 a reference line drawn in the application also records its two
endpoints, in source pixels, as `reference_line`: `[[x1, y1], [x2, y2]]`.
The key is written only when the endpoints are known, so configurations
without it are byte-identical to what earlier versions wrote, and older
configurations that hold only a length still load. The length always equals the
distance between the endpoints; a file whose endpoints disagree with its stored
length keeps the length and ignores the endpoints.

Endpoints let the calibration follow a template that is aligned to another
camera placement or resolution (see the quickstart): the aligned configuration
for each video carries transformed ROI polygons (rounded to whole pixels),
transformed endpoints, a recomputed `_reference_length_pixels`, and
`source_dimensions` set to that video's frame size. When only a length is
stored, the length is scaled by the alignment's scale factor and
`_reference_name` gets the suffix ` (approximate: scaled by alignment)`.
Alignment is applied before the run, so analysis outputs only ever see the
aligned ROIs; results of an unaligned run are unchanged.

## Configuration schemas

A saved template, batch plan or `run_plan.json` holds the open-field arena
polygon in the `roi_manager` block (key `arena`, ROI schema 3) and the arena
center rule in the paradigm parameters (`arena_mode`,
`arena_center_mode`, `arena_center_distance_cm`,
`arena_center_area_fraction`; batch settings schema 6). They are written only
when an arena is set, so configurations without an arena are byte-identical to
those of earlier versions, and an older version rejects a configuration with an
arena rather than silently ignoring it.

## Checkpoints

Resumable runs write to a `.checkpoints/` folder in the output directory, two
files per video:

- `<video>_<id>_checkpoint.json`: a small header (checkpoint version, settings
  hash, video fingerprint, last processed frame, temporal engine state, scalar
  results and the number of committed rows), rewritten atomically;
- `<video>_<id>_checkpoint.rows.jsonl`: one JSON line per processed frame,
  appended and flushed before the header is rewritten.

The checkpoint format is version 4 (unchanged in 3.3 and 3.4). The run folder's
`run_plan.json` is what lets a checkpoint be resumed after the application was
restarted (see "Run plan" above). Checkpoints from other versions, or whose
settings or video do not match, are ignored and the run starts fresh. Both
files are removed when a video finishes.
