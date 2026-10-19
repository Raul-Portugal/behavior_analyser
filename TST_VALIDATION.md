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

## Reading the exported evidence and comparing immobility time

Validation can run after an analysis, without the in-memory result:

- `load_tst_evidence_csv(path)` reads `<base>_tst_motion_evidence.csv` into the
  per-mouse structure that `evaluate_annotations` expects. The CSV `frame`
  column (the absolute video frame) becomes `source_frame`, so annotations
  line up even when the run did not start at frame 0.
- `immobility_time_agreement(evidence, annotations)` compares automated and
  manual immobility seconds per mouse over the annotated frames (frames
  annotated `uncertain` are left out of both sides). Annotate every frame of
  the scored window if you want the seconds to equal total immobility time.
- `immobility_time_pairs(auto_totals, manual_totals)` pairs per-mouse totals
  (for example `TSTResult.total_immobility_time` against stopwatch scoring).
  Pass either pair list to `core.agreement_stats.summarize_agreement` for
  ICC(2,1), Bland-Altman, Pearson and Spearman across mice or videos.

For how to choose the validation set, blind the scoring, which statistics to
report and pre-registered acceptance criteria, see
[`VALIDATION_PROTOCOL.md`](VALIDATION_PROTOCOL.md).
