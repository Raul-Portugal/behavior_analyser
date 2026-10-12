# TST motion-decomposition engine

The Tail Suspension Test uses `tst_motion_decomposition_v3`. The former
frame-difference, v1 single-frame and v2 frame-rate-dependent classifiers have
been removed from the analysis path. Saved configurations, tuner profiles and
checkpoints from earlier engines are rejected with a re-tune message because
their thresholds do not have a scientifically defensible automatic migration.

## Version 3 changes

### Frame-rate normalization

Every motion feature is computed per frame, so v2 thresholds meant different
things at different frame rates. v3 expresses all features relative to a
30 fps reference (`REFERENCE_FPS`), so the default thresholds keep their v2
meaning at 30 fps and the same physical motion scores alike at 15, 30 and
60 fps:

- displacement-type features (`rigid_motion`, `rotation_motion`,
  `residual_motion`, `rotation_radians`, `rigid_dx`/`rigid_dy`) are multiplied
  by `fps / 30`;
- `deformation` is a per-frame silhouette change (IoU loss) and is treated as
  displacement-type, also multiplied by `fps / 30`;
- `motion_injection` and `angular_injection` are differences of per-frame
  velocities (roughly accelerations) and are multiplied by `(fps / 30)^2`;
- ratios (`coherence`, `affine_inlier_ratio`, `mask_area_ratio`) and pendulum
  ranges and amplitudes of position are already rate-invariant. The per-frame
  velocity floor used to count reversals is compared at the reference rate.

The values written to the motion-evidence CSV, the JSON results and the tuner
are these normalized values, not raw per-frame values. Because measurement
noise is also scaled, features at frame rates far above 30 fps are slightly
noisier than at 30 fps; deformation of fast, large limb movements saturates at
low frame rates, so it agrees across rates only approximately.

### Canonical analysis geometry

The engine resizes each frame to `analysis_width` (default 480, saved in the
`motion_config`; frames narrower than this are never upscaled) with area
interpolation before any processing, and scales the ROI polygons to match.
Gray conversion, background stabilization, dense optical flow and segmentation
run only inside the bounding box of the union of all ROIs expanded by 15% of
its size (clamped to the frame). Camera-motion features come from background
inside that box but outside the ROIs; if too few are found the frame pair is
analysed without stabilization. Body-size-normalized features and the pixel
parameters of the feature trackers therefore no longer depend on the source
resolution. Mask contours exported for overlays are mapped back to source
pixel coordinates.

The tuner passes source frames and source-space ROIs to the same engine path,
so its reduced analysis rate (about 12 fps) is a legitimate approximation of
the native-rate run rather than a systematic bias. On a synthetic 1080p frame
pair the per-frame cost fell from about 460 ms to about 14 ms.

### Gap handling

When a mouse is unscorable because segmentation failed, too few mouse pixels
were found, or the rigid fit failed or was weak, that mouse's temporal history
(rigid velocity, angular velocity and every pendulum trace) is cleared. The
next valid frame therefore reports zero injection and `temporal_ready = false`
until enough samples accumulate again, and a pendulum window never spans a gap.

### Window coverage

If `per_roi_times` requests a window beyond the analysed frames, the summary
now reports the window actually scored (`window_start`, `window_end`) next to
the requested one (`requested_window_start`, `requested_window_end`). A request
exceeding the covered window by more than one frame plus 0.5 s sets
`window_truncated` in the quality metrics and raises a quality warning.
`bouts_split_by_uncertainty` counts immobile bouts separated only by uncertain
frames, which may really be one bout.

## What the engine measures

For each mouse and frame pair, the engine:

1. estimates and compensates for camera motion using background features;
2. segments the mouse inside its ROI using local background contrast;
3. measures optical flow on the mouse;
4. fits a robust rigid translation/rotation model using feature tracking and
   a shared dense-flow field;
5. separates rigid whole-body motion from residual articulated motion;
6. measures silhouette deformation, motion coherence, linear and angular
   motion injection;
7. accumulates a bounded temporal history and measures reversals,
   periodicity, dominant-axis motion, cleanliness, and amplitude; and
8. emits a behavioral state, confidence, and QC reason.

The possible raw states are:

- `immobile_static`
- `passive_swing`
- `active_escape`
- `uncertain`

Passive swing is assigned only after the configured observation duration and
when the trajectory supplies pendulum-like temporal evidence. A one-way rigid
translation is not automatically called passive. Rotation is included, so a
rigid mouse swinging around its suspension point can become passive even when
its centroid barely moves. Non-rigid movement, substantial deformation, or
abrupt linear/angular injection is mapped to active escape. Weak segmentation,
insufficient temporal context, and conflicting evidence remain uncertain and
are excluded from the scored-time denominator.

## Temporal scoring

Static immobility and passive swing are combined into protocol immobility.
Short active gaps bounded by immobility may be bridged, and bouts shorter than
the configured minimum duration are removed. Uncertain frames always break a
bout and are never silently interpreted as either mobility or immobility;
`bouts_split_by_uncertainty` reports how many bout boundaries are caused by
uncertainty alone.
Summaries report bridged-gap immobility separately so static, passive-swing,
and bridged components add up to the final immobility total.

## Quality review

The frame-level motion CSV contains:

- active-motion evidence;
- raw behavioral state and final immobility mapping;
- confidence;
- rigid translation, angular motion, and residual motion;
- deformation and coherence;
- linear and angular motion injection;
- temporal pendulum score and readiness;
- segmentation area and affine-fit quality; and
- a QC reason.

Runs with more than 5% uncertain TST time produce a quality warning. The
validation video distinguishes static immobility, passive swing, active escape,
and uncertain periods.

The interactive tuner is intentionally a fast review, not a miniature final
analysis: it samples short start, middle, and end sections at a reduced frame
rate through the same normalized engine. The white body outline is the segmentation mask actually supplied
to the motion classifier. The full analysis always uses every selected source
frame at its native frame rate (internally analysed at the canonical width). Advanced settings can be saved as an
engine-versioned JSON profile; they should be changed only with a documented
validation rationale.

The ordinary analysis summary and batch summary also include mask-area
variation, segmentation-failure percentage, low-confidence percentage, and
state switches per minute. These are quality-review signals, not corrections
to the behavioral outcome.

## Scientific status

Synthetic regression tests verify rigid rotation/translation, non-rigid
deformation, monotonic motion rejection, multi-mouse flow reuse, and
checkpoint-resume equivalence. This is engineering verification, not criterion
validation. Before treating the engine as a validated scientific scorer, it
must still be compared against blinded, frame-accurate expert annotations on
representative recordings and an independent camera/setup holdout.
