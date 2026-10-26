# Roadmap and status

This is the single source of truth for **where the project stands and what comes next**.

- Read it at the start of any planning session.
- Update it at the end of every task that changes status.
- Keep it short. History goes in `CHANGELOG.md` and git, not here.

_Last updated: 2026-10-26 (after the 4.0.0 release)._

## Current status
| Version | State |
|---|---|
| 3.2.0 – 4.0.0 | Released (GitHub releases with exe). |

4.0.0 (single-window workspace plus the guided six-stage UX) was merged in PR #6 (`00190fc`) and released as `v4.0.0` with the exe and its `.sha256`. The release went out before a manual GUI pass was recorded.

### Next step
1. **Maintainer manual GUI pass** on a real display against the released 4.0.0. Walk a Y-maze study from Videos to Results with per-video starts and one adjusted video. Use the checklist in `quickstart.md`. Agents can only test offscreen. Fix anything found as 4.0.x, one conversation per issue.
2. Start TST engine v4 (below), Phase 0.

## Open follow-ups (small; one session each)
- [ ] Delete merged branches, local and remote (everything except `main` and `fix/tst-engine-v3`). The stale `.claude/worktrees/*` are already removed. Auto mode refuses bulk branch deletion, so the maintainer has to run the commands.
- [ ] `fix/tst-engine-v3` is the only branch not merged into `main` (9 commits, local and `origin`). Check whether it was superseded before deleting.
- [ ] Paradigm cards clip their description at very narrow widths (Videos stage).
- [ ] A Detection "reviewed" flag exists (`Project.detection_reviewed`). Confirm the UX feels right after the manual pass.
- [ ] Classic UI (`--classic`): decide when to retire it (keep at least through 4.x).
- [ ] At 1366×768 the Review & run page scrolls slightly when problems are listed.
- [ ] `gui/workspace/setup_tab.py` is no longer used by the UI. It is kept only as a reference in equivalence tests; decide whether to delete it along with those tests.

## Planned: TST engine v4 (target 4.1.0)
Replace the v3 TST classifier with a rigid-motion-compensated, body-core activity engine. It has four plain-language parameters, instant re-scoring and "calibrate to my scoring". The full plan, phases and benchmark data spec are in **`docs/TST_V4_PLAN.md`**.
- [ ] Phase 0: benchmark data (maintainer scores videos) + harness + v3 baseline. **Next.**
- [ ] Phases 1–6: see the plan.

## Backlog / ideas (not scheduled)
- **Criterion validation run.** Score real videos blind with the Validation menu and build the agreement report. This is the most important *scientific* task left (see `VALIDATION_PROTOCOL.md`).
- Exact-only performance: DIS flow or luma-only decode were rejected because they change numbers. Revisit only with a validation study.
- More paradigms (FST reusing the TST engine, NOR, light-dark, 3-chamber), if the lab needs them.

## Standing decisions (from the maintainer)
- No DeepLabCut/SLEAP. Keep the built-in detector.
- Speed-ups must be **exact**: no change to numbers vs earlier releases.
- UI direction: guided single-window workspace with full-window stages, live edits, plain language, light/dark design system.
- Per-video trial start/end and per-video zone adjustment are first-class (template + overrides model, project schema 2).
- Work is phased so a 5-hour usage cutoff never loses work: commit and push after every task.

## Version history (one line each; details in CHANGELOG.md)
| Version | Contents |
|---|---|
| 3.2.0 | Correctness fixes: tuner/production parity, TST v3 (fps-normalized), calibration, crash handling. |
| 3.3.0 | Exact performance gains (TST about 4×), usability quick wins, release automation. |
| 3.4.0 | Parallel batches, CLI, open-field arena/thigmotaxis, sample sheet, time bins, ROI alignment. |
| 3.5.0 | Validation suite (agreement statistics, blind manual scoring, report). |
| 4.0.0 | Workspace UI, projects (`.bmaproj`), ROI editor, guided stages, per-video trials and zones. |
