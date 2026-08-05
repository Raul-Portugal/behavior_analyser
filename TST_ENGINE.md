# TST motion-decomposition engine

The Tail Suspension Test uses `tst_motion_decomposition_v2`. The former
frame-difference and v1 single-frame rigid-motion classifiers have been
removed from the analysis path. Legacy TST configurations are rejected because
their thresholds do not have a scientifically defensible automatic migration.

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
bout and are never silently interpreted as either mobility or immobility.
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
analysis: it samples short start, middle, and end sections at a lower preview
resolution. The white body outline is the segmentation mask actually supplied
to the motion classifier. The full analysis always uses every selected source
frame at its original resolution. Advanced settings can be saved as an
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
