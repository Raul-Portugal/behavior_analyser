# TST validation workflow

The TST classifier is a reproducible motion-based method, not a substitute for
validation on the recording conditions where it will be used. Normal analysis
does not generate extra annotation material or per-frame mask files.

To evaluate a fixed engine profile, prepare an independent CSV with these
columns:

```csv
mouse_name,frame,state
mouse_1,120,immobile_static
mouse_1,150,passive_swing
mouse_1,180,active_escape
```

Allowed labels are `immobile_static`, `passive_swing`, `active_escape`, and
`uncertain`. Keep the profile, video frame rate, and scoring protocol fixed.
Use `core.tst_validation.load_annotations` and `evaluate_annotations` against
the saved motion evidence to report exact state agreement plus immobility
precision, recall, and F1. Assess agreement separately for each setup and
report the number of annotated frames; do not generalize a result from one
camera or suspension setup to another.
