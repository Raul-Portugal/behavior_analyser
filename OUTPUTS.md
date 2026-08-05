# Analysis output contract

Each analysis creates one timestamped run folder:

```text
analysis_run_YYYYMMDD_HHMMSS/
  manifest.json
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
| `*_arm_entries.csv` | Y-maze | no | Debounced, center-mediated arm-entry events |
| `*_speed_data.csv` | Freestyle | no | Per-frame speed and immobility classification |

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
explicitly rather than silently inflating totals.

Validation media is optional and lives in `media/`; it is not created merely
because plots or tabular outputs are generated.

For TST, the existing summaries include quality-control fields for body-mask
stability, segmentation failures, low-confidence frames, and state switching.
They are deliberately not written as an additional per-video QC file.
