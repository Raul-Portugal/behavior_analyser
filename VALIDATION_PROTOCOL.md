# Criterion validation protocol

This protocol describes how a lab can check the automated scores of Behavioral
Maze Analyzer against blinded manual scoring and report the result. It
supports the Y-maze, elevated plus maze (EPM), open field (Freestyle) and tail
suspension test (TST).

Validation is a statement about a **fixed configuration on a defined set of
recordings**: one software version, one set of detection settings, ROIs and
(for TST) one tuned profile, applied to the setups, lighting and animals in the
validation set. It does not transfer to a different camera, arena, lighting or
strain without repeating it there.

The computation layer is in `core/agreement_stats.py` (ICC, kappa,
Bland-Altman, correlation, error metrics, batch summary),
`core/positional_validation.py` (Y-maze, EPM, frame labels, positions) and
`core/tst_validation.py` (TST frames and immobility time). All of it is pure
Python and NumPy and never changes an analysis output. For TST frame
annotations see also `TST_VALIDATION.md`.

## 1. Before you start

1. **Tune first, validate later.** Choose detection settings, ROIs, the TST
   profile and every threshold using a *tuning set*. Then freeze them. Any
   change after validation scoring has started means the validation must be
   repeated.
2. **Record the configuration.** Keep the application version, the saved
   settings (the run's `manifest.json` and `*_summary.json` list the parameters
   used) and the ROI/arena files with the validation results.
3. **Pre-register the endpoints and acceptance criteria** (section 5) before
   anyone scores a video. Write them down with a date.

## 2. Choosing the validation set

- **Held out.** No validation recording may have been used to tune or
  troubleshoot the detection, ROIs or TST profile. If a video was ever
  inspected while tuning, it belongs to the tuning set.
- **Representative, not cherry-picked.** Draw the recordings by a documented
  rule (random, or random within strata) from everything you intend to analyse
  with this configuration. Do not leave out recordings because they look hard;
  hard recordings are exactly where agreement can fail.
- **Cover the sources of variation.** Stratify, so that each of these is
  represented in the set in rough proportion to its use:
  - camera and arena setups (position, resolution, frame rate, each physical
    apparatus),
  - lighting conditions and recording sessions or days,
  - animals: coat colour, sex, age or cohort, and every experimental group
    (treatment and control), so a bias that differs between groups would show,
  - for TST, each suspension rig and number of mice per frame.
- **Suggested minimum: at least 10 videos per paradigm, preferably 20 or
  more**, and at least a few from each setup. This is a precision argument, not
  a power calculation. With two raters and a true ICC near 0.90, simulations
  with `core.agreement_stats.icc` give a median 95 % confidence interval of
  about 0.62 to 0.97 at n = 10, 0.75 to 0.96 at n = 20 and 0.80 to 0.95 at
  n = 30. At n = 20 the interval can therefore rule out poor agreement
  (lower bound below 0.75) but at n = 10 it generally cannot. Bland-Altman
  limits of agreement are estimated even less precisely: their 95 % interval is
  roughly +/-1.3 SD of the differences at n = 10 and +/-0.8 SD at n = 20. Pool
  more videos if the study will report effect sizes near the measurement
  error.
- **Event paradigms need enough events too.** For event-level precision and
  recall aim for roughly 200 manual events in total per paradigm; with 95 %
  precision that gives about +/-3 percentage points.
- **Frames are not independent.** Neighbouring frames are strongly
  autocorrelated, so frame-level kappa from thousands of frames has a
  misleadingly narrow interval. Treat the *video* (or animal) as the unit for
  inference: report per-video values, and do not read the frame count as a
  sample size.

## 3. Blinded manual scoring

- **Blind to the automated output.** The scorer must not see the application's
  tracking, validation video, plots or summary for the recording, and must not
  have seen its scores while tuning. Give the scorer the raw video file, with a
  coded filename that hides the experimental group where possible.
- **Randomize the order** of videos and score them independently of the
  automated run.
- **Two scorers, ideally.** Have a second scorer score the same recordings
  (all of them, or at least a random 30 to 50 %) independently. Their
  agreement (ICC, Bland-Altman, kappa) is the **inter-rater reference**: the
  automated method cannot be expected to agree with a human better than humans
  agree with each other, and it is the fair benchmark for the acceptance
  criteria. Compare the automated result with each scorer separately, or with
  a pre-agreed consensus scored afterwards, never with the scorer who also saw
  the automated output.
- **Write the scoring rules down** (what counts as an arm entry, when an
  animal is immobile) and train the scorers on non-validation videos.
- **Do not edit scores after unblinding.**

### What to score per paradigm

| Paradigm | Manual scoring | File to produce |
| --- | --- | --- |
| Y-maze | Every arm entry in order with its time (an entry is the body crossing into the arm; use the same rule as the configured center-mediated counting and say whether the initial arm counts) | `entry_number,arm,timestamp_s`, the same columns as `<base>_arm_entries.csv`. Arms may be written `A`, `a`, `arm_a`. |
| EPM | Every arm entry with its time and arm (`open_arm_1`, `closed_arm_2`, or just `open` / `closed`), plus time spent in the open arms | `entry_number,arm,timestamp_s`; open-arm time as a number in seconds per video |
| Open field | Time in the center zone, and total immobility time using a stated rule (for example no displacement of the body center for at least 1 s) | per-video seconds; for frame-level agreement a per-frame zone label |
| TST | Immobility. Either total immobility seconds per mouse (stopwatch or event scoring software), or frame / bout annotations | totals per mouse, or `mouse_name,frame,state` as in `TST_VALIDATION.md`. Annotate every frame of the scored window when you want seconds derived from frames. |

Score the same time window the analysis scores (for TST, the configured scoring
window per mouse). For TST, note that `immobile_static` and `passive_swing`
both count as immobile.

### Primary and secondary endpoints

Pick one **primary endpoint** per paradigm before scoring, the measure your
study will report:

- Y-maze: alternation percentage (and total entries as a check on
  activity).
- EPM: percent time in open arms and number of open-arm entries.
- Open field: center time (and immobility time if reported).
- TST: total immobility time per mouse.

Other quantities (event timing, frame labels, position error) are secondary.

## 4. Running the comparison

### In the application (Validation menu)

Everything below can be done without leaving the application. The
**Validation** menu has three tools; their statistics are exactly the ones
described in this protocol (they call the same functions), and none of them
changes an analysis output.

1. **Validation > Score a video manually...** (blind scoring). Choose the
   paradigm, the video, your initials and where to save. You may also load the
   template or batch plan to draw the zone outlines (only the outlines you
   drew; nothing from the analysis is loaded, so the scoring is blind). The
   video player plays at 0.25x to 2x, steps one frame (Left/Right), jumps 10
   frames (Shift+Left/Right), scrubs, and goes to the start/end with Home/End;
   Space plays and pauses. Hotkeys mark the frame on screen:

   | Paradigm | Keys |
   | --- | --- |
   | Y-maze | `A`, `B`, `C` (or `1`, `2`, `3`) log an entry into that arm; `0` marks the center (optional, not an entry) |
   | EPM | `1` open arm 1, `2` open arm 2, `3` closed arm 1, `4` closed arm 2, `0` center |
   | Open field | `C` toggles center / periphery, or, in hold mode, is held while the animal is in the center |
   | TST | hold `I` immobile, `P` passive swing (counts as immobile), `A` active, `U` uncertain; every frame shown while the key is down gets that state |

   Ctrl+Z undoes, Delete removes the selected row, and double-clicking a row
   jumps to its frame. Progress is autosaved next to the output files
   (`<video>_manual_scoring_<paradigm>_<initials>.json`, which also records
   the scorer's initials, the date, the application version and that the
   session was blind) and an interrupted session resumes when you reopen the
   same video with the same initials. **Save** writes:

   | File | Content |
   | --- | --- |
   | `<video>_manual_arm_entries.csv` | `entry_number,arm,timestamp_s`, the columns of `<base>_arm_entries.csv` (Y-maze `arm_a`..; EPM `open_arm_1`..) |
   | `<video>_manual_epm_totals.csv` | EPM only, when you marked the center: `metric,value` with `time_in_open_arms_s`, `time_in_closed_arms_s`, `time_in_center_s`, `percent_time_in_open`. Each mark lasts until the next one (or the end of the scoring window); time before the first mark is not counted |
   | `<video>_manual_zone_intervals.csv` | Open field: `interval_number,zone,start_frame,end_frame,start_s,end_s,duration_s`. Intervals tile the scoring window; frames are inclusive, `start_s = start_frame / fps`, `end_s = (end_frame + 1) / fps` |
   | `<video>_manual_tst_annotations.csv` | `mouse_name,frame,state` as in `TST_VALIDATION.md`. Choose the mouse in the dialog; with "unmarked frames = active_escape" every frame of the scoring window is labelled, so immobility seconds can be derived |

   Set the scoring window (start and end) to the window the analysis scores.
   A second scorer should save to a different folder; the dialog asks before
   replacing files that this session did not write.

2. **Validation > Tracking spot-check...** (position accuracy). Pick a run
   folder (or a video and its `*_tracking_data.csv`). The dialog draws N random
   frames (default 50) with a visible fixed seed (the same seed on the same
   tracking file gives the same frames), shows each *without* the automated
   dot, and you click the animal's center or press **Not visible** (N);
   Backspace goes back. At the end it reports position error in px and cm
   (using the run's `scale_factor`), the miss rate and false detections, and
   saves `<base>_position_spotcheck.csv` and
   `<base>_position_spotcheck_summary.json` (seed, sampled frames, counts,
   statistics, scorer and date).

3. **Validation > Build validation report...** (agreement report). Add the run
   folder(s) and the folder(s) with the manual files; they are matched per video
   by filename stem (a manual file called `video1_manual_arm_entries.csv`
   matches the run output of `video1`; other CSVs are recognised by their
   header). Review the matching table, fix it where needed (double-click a
   cell to choose another file), then set the event tolerance, the
   acceptance criteria and the output folder. The report is one HTML file
   (all styles and plots embedded, nothing loaded from the network) plus a
   CSV of every table. It contains the per-video tables, pooled agreement
   (ICC(2,1) with CI, Bland-Altman bias and limits of agreement, Pearson and
   Spearman), event precision, recall and F1, Bland-Altman and scatter plots,
   the TST (and open-field zone) confusion matrix with kappa, the position
   spot-check, the acceptance criteria marked pass or fail, and the
   application version, date and every input file with its SHA-256.

   Event precision, recall and F1 are pooled over videos by summing the
   counts, and the mean of the per-video F1 is shown beside the pooled value.
   Open-field center time needs a run with the **open-field arena** option
   (`freestyle_metrics.arena` in the summary). Criteria that cannot be
   computed (no data, or fewer than two videos) are marked "not evaluable",
   never silently passed.

The Python flow below does the same per video if you prefer scripts.

### From Python

Typical flow after the analysis has finished (per video):

```python
from core.positional_validation import (
    load_entry_annotations, load_tracking_csv, compare_y_maze,
    summarize_agreement,
)

auto = load_entry_annotations("out/video1_arm_entries.csv", "y_maze")
manual = load_entry_annotations("manual/video1.csv", "y_maze")
per_video = compare_y_maze(auto, manual, tolerance_s=1.0)

# After all videos: one (auto, manual) pair per video for the endpoint.
pairs = [(c.auto_alternation_pct, c.manual_alternation_pct) for c in comparisons]
summary = summarize_agreement(pairs)      # ICC(2,1), Bland-Altman, r, rho, RMSE...
```

For TST use `core.tst_validation.load_tst_evidence_csv` on
`<base>_tst_motion_evidence.csv`, `immobility_time_agreement` for frame
annotations or `immobility_time_pairs` for stopwatch totals, then
`summarize_agreement`. Pairs with a missing value are excluded and counted
(`n_excluded`); report that number.

Choose the matching tolerance (`tolerance_s`) in advance. A tolerance of about
1 s is a reasonable starting point for arm entries, and 2 s or more is
appropriate when the scorer works from a normal-speed video without frame
stepping; state the value you used.

## 5. Statistics to report

Report for each primary endpoint, per setup if the set is large enough, and
for the inter-rater comparison when two scorers exist:

| Statistic | Function | Use |
| --- | --- | --- |
| **ICC(2,1)** with 95 % CI | `icc(..., "2,1")` | Absolute agreement of a single automated reading with a single manual reading. The main criterion. Use ICC(3,1) only to show how much of a disagreement is a constant offset. |
| **Bland-Altman** bias, SD of differences, 95 % limits of agreement with their CIs | `bland_altman` | Size and direction of the error in the measurement's own units. A positive bias means the automated score is higher. |
| Proportional bias | slope and its p-value in `bland_altman` | Whether the error grows with the magnitude. |
| Pearson r and Spearman rho with CI | `pearson`, `spearman` | Supporting only. Correlation ignores bias and scale, so a high r does not show agreement. |
| RMSE, MAE, mean absolute percentage difference | `rmse`, `mae`, `mean_absolute_percentage_difference` | Typical error size, in units and in percent. |
| **Cohen's kappa** and percent agreement | `frame_label_agreement`, `cohens_kappa` | Frame-level labels (zones, TST states). Report together with the label counts, because kappa falls when one label dominates. |
| **Event precision, recall, F1** and mean timing offset | `match_events`, `compare_y_maze` | Arm entries: precision = share of automated entries that are real, recall = share of real entries found. State the tolerance. |
| Sequence similarity | `sequence_edit_distance` | Y-maze entry sequences (alternation depends on order, not only counts). |
| Position error | `position_errors` | Mean, median and 95th percentile in px and cm, the detection miss rate and false detections. Only for frames the scorer reviewed. |

### Interpreting the results

- **ICC** (Koo and Li, 2016, as a rule of thumb for the *confidence interval*,
  not just the estimate): below 0.50 poor, 0.50 to 0.75 moderate, 0.75 to 0.90
  good, above 0.90 excellent. Always give the CI; a high estimate with a wide
  CI (small n) is weak evidence.
- **Kappa** (Landis and Koch, 1977): 0.61 to 0.80 substantial, above 0.80 almost
  perfect. Treat these as conventions rather than thresholds, and report the
  percent agreement beside it. Kappa is not reported (nan) when both raters
  use a single label.
- **Bland-Altman:** judge the limits of agreement against the *effect you want
  to detect*, not against a statistical test. If 95 % of automated readings
  can differ from manual by up to +/-10 percentage points of alternation, ask
  whether treatment effects of that size matter in your study. A bias interval
  that excludes 0 is a systematic offset; it can be acceptable if small and
  stable, and it argues against pooling automated and manual data.
- **Event metrics:** high recall with low precision means spurious entries
  (often ROI jitter at a boundary); high precision with low recall means missed
  entries (often tracking gaps; see `entries_possibly_missed`).
- **Compare with the inter-rater reference** whenever two scorers exist: if
  two humans reach ICC 0.85, an automated ICC of 0.85 against either of them is
  at human level.

### Acceptance criteria to pre-register

State these before scoring. A reasonable default, to be tightened or relaxed
with a written reason:

1. **Primary endpoint:** ICC(2,1) point estimate **>= 0.90**, with the lower
   bound of the 95 % CI **>= 0.75**.
2. **Bias:** |Bland-Altman bias| below a stated tolerance in the endpoint's
   units, chosen from the smallest group difference you care about. Examples:
   alternation within +/-5 percentage points; percent time in open arms within
   +/-3 percentage points; TST immobility within +/-5 s over a 240 s window.
   Also state the acceptable width of the limits of agreement.
3. **No proportional bias** at the 5 % level, or a stated reason why it is
   acceptable.
4. **Event paradigms:** event F1 >= 0.90 at the stated tolerance, with
   precision and recall each >= 0.85.
5. **Frame labels:** kappa >= 0.80 (TST: also immobility precision and recall
   >= 0.85).
6. **Not worse than the inter-rater reference** when two scorers were used (the
   automated ICC against each scorer is within the CI of the scorer-to-scorer
   ICC).
7. **Per setup:** the criteria hold in each setup individually, or the setups
   that fail are excluded from automated use.

If a criterion is missed, report it. The usual options are to retune on a
*new* tuning set and validate again on a *new* held-out set, to restrict the
automated analysis to the setups that passed, or to keep manual scoring for
the endpoint.

## 6. Methods-section template

Edit the bracketed parts and delete anything that does not apply.

> **Automated scoring and criterion validation.** Behavior was scored
> automatically with Behavioral Maze Analyzer [version, e.g. 3.5.0] using
> [paradigm] settings that were fixed before validation (detection [method and
> thresholds], [ROI/arena definition], [TST profile and engine
> `tst_motion_decomposition_v3`]). Automated scores were validated against
> blinded manual scoring in a held-out set of [n] recordings ([n] animals,
> [setups/lighting/groups covered]) that was not used for parameter tuning.
> [Scorer 1] [and scorer 2] scored [endpoint, e.g. arm entries and their times
> / total immobility time] from the raw videos without access to the automated
> output [and in randomized order, blind to group]. Agreement for
> [primary endpoint] was assessed with the two-way random-effects, absolute
> agreement intraclass correlation coefficient ICC(2,1) with 95 % confidence
> interval, and with Bland-Altman analysis (bias and 95 % limits of agreement
> with their confidence intervals, and regression of the difference on the
> mean for proportional bias). Event-level agreement used one-to-one matching
> of automated and manual events within [tolerance] s (precision, recall and
> F1) [and Cohen's kappa for frame-level [zone/behavior] labels]. The
> acceptance criteria, specified before scoring, were ICC(2,1) >= [0.90]
> (lower 95 % CI bound >= [0.75]) and an absolute bias < [tolerance, units].
> The automated and manual scores agreed with ICC(2,1) = [value] (95 % CI
> [low, high]); the bias was [value] [units] (95 % limits of agreement [low]
> to [high]) [event F1 = [value] ...]. [Inter-rater agreement between the two
> scorers was ICC(2,1) = [value] (95 % CI [low, high]).] [State which criteria
> were met and any exceptions.] Statistics were computed with the validation
> module distributed with the software (pure Python/NumPy implementations;
> the incomplete-beta based t and F distributions were checked against
> published tables).

## 7. Reporting checklist

- Software version, settings and ROI files; date configuration was frozen.
- How recordings were selected; counts per setup, lighting, group.
- Scorer training, blinding, order, number of scorers.
- Pre-registered endpoints, tolerances and acceptance criteria.
- n videos used, n excluded and why (`n_excluded`).
- ICC(2,1) with CI, Bland-Altman bias with LoA and their CIs, event
  precision/recall at the stated tolerance, kappa with percent agreement.
- Which criteria were met, per setup, and what was done about those that were
  not.
- Raw per-video (automated, manual) values, so others can recompute.
