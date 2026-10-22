# Roadmap and status

This is the single source of truth for **where the project stands and what comes next**.

- Read it at the start of any planning session.
- Update it at the end of every task that changes status.
- Keep it short. History goes in `CHANGELOG.md` and git, not here.

_Last updated: 2026-10-22._

## Current status
| Version | State |
|---|---|
| 3.2.0 – 3.5.0 | Released (GitHub releases with exe). |
| **4.0.0** | **In PR #6 (`release/4.0.0` → `main`), not merged.** Single-window workspace plus the UX redesign (guided six-stage flow). |

### Next step
1. Finish the 4.0 UX redesign, Wave D (Review & run, Results stages, polish, docs). Branch: `feat/stage-review-results`.
2. **Maintainer manual GUI pass** on a real display. Use the checklists in PR #6 and in `quickstart.md`. Agents can only test offscreen.
3. Merge PR #6, then `gh release create v4.0.0 --target <merge sha>` with the notes from the 4.0.0 section of `CHANGELOG.md`.

## Open follow-ups (small; one session each)
- [ ] Delete merged branches (`fix/*`, `feat/*`, `chore/*`, old `release/*`) and stale `.claude/worktrees/*`. Auto mode refuses to delete these, so the maintainer has to run the commands.
- [ ] Paradigm cards clip their description at very narrow widths (Videos stage).
- [ ] A Detection "reviewed" flag exists (`Project.detection_reviewed`). Confirm the UX feels right after the manual pass.
- [ ] Classic UI (`--classic`): decide when to retire it (keep at least through 4.x).

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
