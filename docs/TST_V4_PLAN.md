# TST engine v4: implementation plan

_Written 2026-10-26. Owner: maintainer + Claude sessions. Status: **planned, not started**._

This is the working plan for replacing the Tail Suspension Test classifier
(`tst_motion_decomposition_v3`) with a simpler, more accurate engine. Each phase
below is sized for **one conversation**. Tick items here and keep
`docs/ROADMAP.md` pointing at the current phase.

## 1. Goal

Classify **immobility vs mobility** in the TST so that it agrees with expert
human scoring, runs comfortably on a laptop, and exposes a few plain-language
parameters the user can change and see the effect of instantly.

Required behaviour:

| Situation | Must be scored as |
|---|---|
| Hanging still (breathing only) | immobile |
| Passive swinging / pendulum motion, including momentum after a struggle | immobile |
| Movement of the paws/limbs only, trunk not engaged | immobile |
| Short struggle shorter than the user's "ignore struggles shorter than" | immobile |
| Sustained struggling: trunk bending/curling, twisting, climbing toward the tail | mobile |
| Mouse not visible / tracking lost | uncertain (excluded, reported) |

Open definition question (see section 9): how passive **spinning** around the
tail axis is scored. Default assumption: passive spin = immobile.

## 2. Why v3 fails (summary of the diagnosis)

- **Paw movement is called active.** The articulated-motion feature is the
  90th percentile of non-rigid motion over the whole outline
  (`core/tst_motion_engine.py`, `_fit_motion`). About 10% of the mouse's pixels
  moving about 1 px per frame (at the 480 px analysis width) crosses
  `active_residual_threshold`. Paws, tail and fur edges reach that easily.
  Outline-change IoU loss reacts to paws the same way.
- **Swing is immobile only if a pendulum detector confirms it.** Without enough
  reversals and periodicity in a 2.5 s window, rigid motion becomes:
  - `active_escape`, because the acceleration at each turnaround trips the
    injection thresholds (a ±12° swing at 2 Hz already exceeds
    `active_angular_injection_threshold`); or
  - `uncertain`, which drops the frame from the denominator.

  Fast swings also distort the flow at the edges, which inflates the residual.
  Spinning in 3D is not rigid in 2D at all.
- **Frames are decided one at a time,** with a 0.2 s gap bridge that uncertain
  frames always break. A 0.4 s struggle therefore ends an immobility bout.
- **There are 16 opaque thresholds,** and none was fitted to human labels.

## 3. Design of v4

### 3.1 Principle
1. Remove whole-body (rigid) motion.
2. Measure only what is left, weighting the **body core** far above the
   extremities.
3. Integrate that over a short time window.
4. Decide with hysteresis and duration rules.

Swinging is cancelled by construction, so no swing detector has to "prove"
anything. Paw movement is down-weighted by construction.

### 3.2 Per-frame feature extraction (the video pass; the expensive part)
For each mouse ROI and each consecutive frame pair:

1. **Segment** the mouse. Start from the v3 segmentation (local background
   contrast, Otsu, morphology, choosing the component near the previous
   centroid), which is reported to work. Add a temporal consistency check on
   area and centroid jumps.
2. **Rigid alignment.** Estimate the previous→current rigid transform of the
   mouse with sparse feature tracking + RANSAC (as in v3), then warp the
   previous mouse crop into the current pose. Work only in the mouse bounding
   box (plus a margin), not the whole ROI box.
3. **Core weighting.** Compute the distance transform `D` of the current mask
   and normalise it by the body half-width (the maximum of `D`). The weight is
   `w = clip(D / (k · max D), 0, 1)` with `k` set by the limb setting:
   - Off: `k → 0`, uniform weights.
   - Normal and Strong: increasing `k`.

   Thin protrusions (paws, tail, fur edge) get weight ≈ 0. Trunk pixels get
   weight ≈ 1.
4. **Candidate activity features.** Phase 1 picks one or two of these using
   data:
   - **F1 aligned residual:** `Σ w·|I_t − warp(I_{t−1})| / Σ w`, normalised by
     the mouse/background contrast.
   - **F2 residual flow:** dense flow inside the mouse box (Farneback or DIS)
     minus the rigid prediction, then the core-weighted mean magnitude
     divided by the body size.
   - **F3 posture change:** rate of change of body length/curl (major axis or
     skeleton length, head-to-hook distance) and core-weighted IoU loss after
     alignment.
   - **F4 rigid motion magnitude:** never used to decide. It is recorded for
     QC and for reporting "swing" time.
5. **Normalisation.** Divide by body size, and express rates per second rather
   than per frame, so 15/30/60 fps and different resolutions score alike.
   Reuse the v3 canonical geometry: `analysis_width`, crop, exact-crop
   factors.
6. **Store raw per-frame values** for all three limb settings (Off / Normal /
   Strong) so the limb control re-scores without re-reading the video.

The engine keeps **no temporal state except the previous frame**. Everything
temporal happens in the scoring layer, which makes checkpoints and the
threaded `TSTFramePipeline` simpler than in v3.

### 3.3 Scoring layer (pure functions; instant)
Input: the per-frame activity series of one mouse plus the four user
parameters. Output: per-frame `immobile / mobile / uncertain` plus bouts.

1. **Smoothing:** centred moving mean over `W` seconds. `W` is a fixed
   constant, about 0.5 s, chosen in Phase 1.
2. **Hysteresis:** a bout becomes mobile when activity rises above `T` and
   ends when it falls below `h·T`. `h` is a fixed constant, about 0.6, chosen
   in Phase 1. `T` comes from the Sensitivity control.
3. **Duration rules, in this order:**
   1. mobile bouts shorter than *min mobility* become immobile;
   2. immobile bouts shorter than *min immobility* become mobile.
4. **Gaps:** uncertain (tracking lost) runs up to 0.5 s that have the same
   state on both sides take that state. Longer runs stay uncertain, are
   excluded from the denominator and are reported.
5. **Study-wide threshold.** The threshold is never adapted per video, because
   that would bias comparisons between animals. Normalising for body size and
   contrast is allowed.

### 3.4 User parameters (the only ones in the normal UI)

| Control | Meaning | Range (initial guess; Phase 1 sets defaults) |
|---|---|---|
| **Sensitivity** | How much trunk movement counts as mobility. Shown as a line on the activity trace. | log scale, default 1.0 = fitted default |
| **Ignore limb-only movements** | Off / Normal / Strong | default Normal |
| **Ignore struggles shorter than** | min mobility bout | 0–3 s, default ≈ 1 s |
| **Count immobility only if it lasts at least** | min immobility bout | 0–5 s, default ≈ 1 s |

Everything else is a fixed constant recorded in the output under the engine
ID. There are no "advanced" thresholds in the UI. A developer-only JSON
profile is allowed for validation work.

### 3.5 Instant re-scoring
- The four controls act on stored per-frame signals, so changing them in the
  Detection stage tuner or the Results stage re-scores in milliseconds without
  re-reading the video.
- The parameters are saved in the project and written to every summary, so
  results stay reproducible.
- **Settings hash:** the feature-extraction config feeds the checkpoint/settings
  hash; the four scoring parameters must **not**, so changing them never
  invalidates checkpoints. Check how TST parameters currently reach
  `BatchSettings.to_dict()` and `get_algorithm_parameters()` before changing
  anything (CLAUDE.md invariant).

### 3.6 "Calibrate to my scoring"
The user scores 2–3 videos with *Validation → Score a video manually*. The app
then:
1. grid-searches the four parameters, maximising frame-level agreement (κ) with
   a penalty on immobility-time error;
2. shows leave-one-video-out agreement, so the user sees how well the settings
   generalise; and
3. offers to apply the settings to the study.

This is how users adjust results "to their liking" while staying objective.

### 3.7 Outputs
- Bump the engine ID to `tst_activity_v4`.
- Keep `*_tst_motion_evidence.csv` readable by `core/tst_validation.py` and
  `core/validation_report.py`: keep `frame`, mouse and `state`, and add the
  v4 columns. Decide the state vocabulary in Phase 4. Either:
  - add `immobile` and `mobile` to the valid/immobile state sets; or
  - map to `immobile_static` / `active_escape`.

  Document it in `OUTPUTS.md` in the same change.
- New metrics are additive: `swing_time_s` (immobile with high rigid motion)
  and `short_struggles_ignored`. Do not rename existing summary columns.
- Engine choice: keep v3 selectable through 4.1 for side-by-side comparison
  (decision pending). New studies default to v4 once Phase 6 passes.

### 3.8 Performance budget
- Feature extraction ≤ 5 ms per mouse per frame at 480 px on a 4-core laptop.
  A 6-minute, 30 fps video with 4 mice should finish in about 2 minutes or
  less, limited by decoding.
- Phase 0 measures decode time and v3 cost on the benchmark videos as the
  baseline.
- Re-scoring a 6-minute series takes < 50 ms (it is NumPy only).

## 4. Phases

Each phase runs on its own `feat/tst-v4-*` branch off `release/4.1.0`. Create
`release/4.1.0` from `main` after PR #6 merges, or from `release/4.0.0` if
work starts earlier. Each phase is commit + push at the end, with tests, ruff
and the startup check green.

### Phase 0: Benchmark data and harness
- [ ] **Maintainer:** prepare the benchmark folder and score the dev set
  (section 5). Claude can build the harness in parallel.
- [ ] `tools/tst_benchmark.py`, a CLI. Given a benchmark folder (manifest,
  annotations, evidence CSVs or a way to run the engine), it reports per video
  and overall:
  - frame κ and percent agreement;
  - immobility precision, recall and F1;
  - per-mouse immobility seconds: ICC(2,1), Bland–Altman bias and limits,
    MAE;
  - bout-level F1 with a ±0.5 s boundary tolerance.

  It also reports the inter-rater ceiling where two raters exist, and writes
  a Markdown/CSV report. It reuses `core/tst_validation.py` and
  `core/agreement_stats.py`.
- [ ] Run v3 on the benchmark through the normal analysis path to get the
  **v3 baseline numbers**.
- [ ] Synthetic clips/series for unit tests: still, rigid swing, rigid
  rotation, paw-only motion, trunk curl, short twitch, tracking loss. Extend
  `tests/tst_synthetic.py`.
- **Done when:** the harness reproduces known agreement numbers on synthetic
  labels and the v3 baseline report exists for the dev set.

### Phase 1: Offline prototype and feature selection (scratch/tools only, no app changes)
- [ ] Prototype F1–F4 in `tools/tst_v4_prototype.py`, sharing helpers with
  nothing in `core/` yet.
- [ ] On the **dev set only**: plot features against human labels. Compare
  F1 vs F2 vs F1+F3, and each limb setting, on κ and immobility-time error.
  Check swing (P-labelled) and paw-only frames specifically.
- [ ] Choose the fixed constants (`W`, `h`, core `k` per limb setting,
  normalisation) and the default values of the four parameters by
  leave-one-video-out cross-validation.
- [ ] Measure per-frame cost on a laptop-class CPU.
- [ ] Write the findings (plots, chosen formula, rejected alternatives) to
  `docs/TST_V4_FINDINGS.md`.
- **Done when:** the prototype beats v3 on the dev set by a clear margin,
  approaches the inter-rater ceiling, and meets the performance budget. If it
  does not, stop and revisit the design (options A/D in the original
  analysis) before writing production code.

### Phase 2: Engine core
- [ ] `core/tst_activity_engine.py`: `TSTActivityEngine`, `TSTActivityConfig`
  (feature-extraction config only) and a per-frame observation dataclass. Same
  public shape as `TSTMotionEngine`: `calculate_observations`,
  `get_last_mask_contour`, `export_state` / `restore_state`, and a
  `TSTFramePipeline`-compatible split into a pure stage and an ordered stage.
- [ ] Reuse the canonical geometry and segmentation from v3. Factor shared
  helpers out of `tst_motion_engine.py` **without changing v3 numbers**; the
  v3 equivalence tests must stay green.
- [ ] Tests:
  - synthetic behaviour (swing ≈ 0 activity, paw-only ≈ 0 with Normal, trunk
    curl high);
  - fps and resolution invariance;
  - checkpoint-resume equivalence;
  - pipeline vs sequential byte-identity.
- **Done when:** the engine reproduces the Phase 1 prototype's per-frame values
  exactly (or within a stated float tolerance) on the dev videos.

### Phase 3: Scoring layer
- [ ] `core/tst_scoring.py` with pure functions: smoothing, hysteresis,
  duration rules, gap handling, bouts, totals, latency.
- [ ] A `rescore(evidence, params)` function used by both the analysis and the
  UI.
- [ ] Unit tests for every rule and for the edges: series start/end, all
  uncertain, zero durations.
- **Done when:** the benchmark harness, run through this layer on stored
  evidence, matches Phase 1.

### Phase 4: Integration
- [ ] `mazes/tst.py`:
  - engine selection (v4 default for new studies, v3 kept);
  - parameters (four scoring parameters + feature config);
  - `calculate_metrics` through `core/tst_scoring.py`.
- [ ] Pipeline / batch / CLI / workspace / checkpoints. Make sure the scoring
  parameters stay out of the settings hash. Byte-identity tests: CLI vs
  workspace vs parallel.
- [ ] Evidence CSV and summary columns (section 3.7), the validation video
  overlay, and updates to `OUTPUTS.md`.
- [ ] Project file: store the scoring parameters in the template. Bump
  `SCHEMA_VERSION` only if needed, with a migration and tests.
- **Done when:** the full suite is green and a benchmark run through the real
  app path equals the harness numbers.

### Phase 5: UI
- [ ] Detection stage, TST tuner (embedded) shows:
  - the video with the mask and a core-weight overlay, so the user sees that
    paws are ignored;
  - the activity trace with the threshold line and a coloured
    immobile/mobile/uncertain ribbon;
  - the four controls;
  - immobility seconds updating live on the sampled segments.

  Remove the v3 advanced spin boxes from the normal view.
- [ ] Results stage: the same four controls re-score all videos instantly from
  stored evidence. A clear "parameters changed since run" note, and outputs
  rewritten on request.
- [ ] "Calibrate to my scoring" wizard (section 3.6), going through
  `window.prompts`.
- [ ] Visual QA with `tools/ui_gallery.py` at 1366×768 and 1920×1080, light and
  dark. Canvases stay ≥ 65% of the width.
- [ ] The classic UI (`--classic`) keeps working, with at least the four
  controls.
- **Done when:** a user can tune, run, adjust in Results and calibrate without
  touching an advanced setting.

### Phase 6: Validation, docs and release
- [ ] Run the harness on the **held-out test set**, which was never used for any
  choice. Report v4 vs v3 vs the inter-rater ceiling.
- [ ] Acceptance (from `VALIDATION_PROTOCOL.md`), checked on the test set:
  - ICC(2,1) of immobility seconds ≥ 0.90 (CI low ≥ 0.75);
  - |bias| ≤ 5 s per 240 s;
  - frame κ ≥ 0.80;
  - immobility precision and recall ≥ 0.85;
  - not worse than inter-rater agreement by more than a small margin;
  - meets the performance budget.
- [ ] Rewrite `TST_ENGINE.md`, `TST_VALIDATION.md` and `quickstart.md` (TST
  section), plus `CHANGELOG.md` (4.1.0) and `OUTPUTS.md`.
- [ ] Release 4.1.0.

## 5. Benchmark data the maintainer provides

**Folder** (outside the repo; videos stay where they are):

```
<BENCH>\                         e.g. D:\TST_benchmark\
  manifest.csv                   one row per video (columns below)
  tst_benchmark.bmaproj          a TST study in the app with these videos,
                                 mouse ROIs and trial windows set up
  labels\<INITIALS>\             one folder per rater (the annotation CSV
                                 name has no initials, so raters would
                                 overwrite each other in a shared folder)
      <video>_manual_tst_annotations.csv
      <video>_manual_scoring_tst_<INITIALS>.json
```

`manifest.csv` columns:
`video_path, split (dev|test), rig_or_camera, fps, mice (mouse_1;mouse_2…),
strain_or_coat, notes`.

Treatment or group can go in `notes` **after** scoring, so the scoring stays
blind.

**Videos:** 12–16 full-length videos (or about 30 mouse-sessions). Split them
about 50/50 into `dev` and `test`. Cover:
- every rig/camera/lighting setup in use;
- each number of mice per frame;
- dark and light coats;
- high and low immobility groups;
- the hard cases: heavy swingers, spinners, paw-only movers, tail climbers.

Score the dev half first; the test half can be scored while Phases 2–5 run.

**Second rater:** 3–4 videos (mixed dev/test) scored independently, for the
inter-rater ceiling.

**How to score** (*Validation → Score a video manually* in the current app):
1. Paradigm TST; pick the video; scorer initials; output folder =
   `labels\<INITIALS>\`. Mice = the **same names as the project ROIs**
   (`mouse_1, mouse_2…`); load the ROI/config file to fill them in.
2. Window start/end = the trial window you analyse, usually the whole 6 min.
   "No key held" = **Unmarked frames in the window = active_escape**.
3. Choose the mouse, play at **0.5×**, and **hold**:
   - `I`: completely still (breathing only);
   - `P`: passive, counts as immobile but something moves: swinging,
     momentum, passive spinning, **paws/limbs only**;
   - `U`: cannot judge (hidden, out of view, climbing its tail).
   - Release (no key): struggling with the trunk engaged. **Label every real
     struggle, however short**; the duration rules are fitted afterwards.
4. Fix boundaries with Left/Right frame stepping, Undo, or Delete selected.
5. Repeat for each mouse (choose the mouse, rewind, score). Save.

Caveat: the scoring window is per session, not per mouse. If mice in one video
were suspended at different times, score each mouse in its own session with
only that mouse listed and its own window, saving into
`labels\<INITIALS>\<mouse>\`. Alternatively, ask for the small dialog fix
(a per-mouse window) at the start of Phase 0.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Segmentation fails on some rigs or coats | Phase 0 includes every rig/coat; QC metrics; uncertain only for real failures |
| 3D spinning looks non-rigid | Explicit P labels for spin; Phase 1 checks spin frames; a smoothness feature if needed |
| Defaults overfit the dev set | Held-out test set; leave-one-video-out; few parameters |
| Hold-key scoring lags reactions (~0.2 s) | 0.5× playback; boundary tolerance in bout F1; lag mostly cancels in totals |
| New engine changes published numbers | New engine ID; v3 stays selectable; the release notes say so |

## 7. Not doing
- Deep learning, DeepLabCut or SLEAP (standing decision). No new dependencies
  without asking; the prototype uses OpenCV + NumPy only.
- Per-video adaptive thresholds.

## 8. Session log
| Date | Phase | Result |
|---|---|---|
| 2026-10-26 | Plan | This document. |

## 9. Decisions pending (maintainer)
1. Passive spinning: immobile (default) or mobile?
2. Keep v3 selectable in 4.1 (default yes) or replace it?
3. Are mice in a multi-mouse video always suspended together? (This decides
   whether the dialog needs per-mouse windows.)
4. Benchmark folder path and which rigs/videos.
