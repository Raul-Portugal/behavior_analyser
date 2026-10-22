# Working on this project with Claude Code

How to work on this project efficiently with a Claude Team seat (5-hour and weekly usage windows). The advice follows the official Claude Code docs at code.claude.com (best-practices, costs, prompt-caching, memory, sub-agents, skills); they were checked on 2026-10-22.

## 1. Why one giant conversation is a problem

Your intuition is correct.

**Cost.** Every request re-sends the whole context: system prompt, CLAUDE.md, every earlier message and every tool result (docs: prompt-caching).
- Cached re-reads are billed cheaper, but they still count.
- The docs say a one-line question in a session open all day "still draws usage for the whole conversation" (docs: costs).
- Each tool call is another request carrying the full history.

**Quality.** "LLM performance degrades as context fills… Claude may start forgetting earlier instructions" (docs: best-practices).

**Caching.** The main conversation's cache lives about 1 hour. Switching model or effort mid-session throws the cache away.

**Rule:** one task per conversation, and keep the project's state in files, not in chat history.

## 2. What is now in the repo

| File | Loaded | Purpose |
|---|---|---|
| `CLAUDE.md` | every session (keep it under ~200 lines) | Commands, layout, invariants (exact outputs, Qt rules), git flow, definition of done |
| `.claude/rules/science-core.md` | only when `core/`, `mazes/`, … are touched | Exactness rules for analysis code |
| `.claude/rules/gui-qt.md` | only when `gui/` is touched | Qt lifecycle rules, design system, visual QA |
| `docs/ROADMAP.md` | read on demand (CLAUDE.md tells Claude to) | **Current status, next step, follow-ups, standing decisions** |
| `.claude/agents/test-runner.md` | when delegated | Haiku runs tests, ruff and the startup check, and reports only failures |
| `.claude/agents/ui-reviewer.md` | when delegated | Renders screens to PNG and reviews the UI |
| `.claude/agents/workstream.md` | when delegated | Sonnet in an isolated worktree: one branch, push after each task |
| `.claude/skills/test` | `/test [modules]` | Runs checks through test-runner |
| `.claude/skills/visual-qa` | `/visual-qa [pages]` | Runs a UI review through ui-reviewer |
| `.claude/skills/release` | `/release 4.1.0` (manual only) | Release checklist; asks before merging and publishing |
| `.claude/skills/handoff` | `/handoff` | Updates ROADMAP, commits and pushes, prints a prompt for the next session |

**Why the state lives in the repo.** `docs/ROADMAP.md` replaces the long chat history as the record of where things stand. CLAUDE.md stays small and stable; the roadmap changes often and is read only when needed. Claude's auto-memory (`~/.claude/projects/…/memory`) is machine-local and transcripts are deleted after 30 days, so durable state belongs in the repo.

## 3. Day-to-day workflow

### One conversation per task
A task is a bug fix, a feature, a release or a review.
- Start a new conversation (or run `/clear`) for each new task.
- Name it with `/rename fix-trials-elide`. Resume the *same* task later with `claude --resume <name>`, or from the app's session list.
- Need a quick side answer? Use `/btw …` so it doesn't enter the history.

### Starting prompt templates
- **Bug:** "Bug: <what you did, what happened, what you expected, screenshot/log>. Read docs/ROADMAP.md if relevant. Find the root cause, write a failing test, fix it, run /test."
- **Feature or phase:** "Read docs/ROADMAP.md. Plan <feature> in plan mode. Ask me questions first."
- **Continue:** paste the prompt that `/handoff` printed.

### Ending a session
Run **`/handoff`**. It records status in `docs/ROADMAP.md`, commits, pushes, and gives you a short prompt for the next conversation. Use it whenever you stop, and before the usage limit runs out.

### Compaction
- **Use `/compact <what to keep>` only mid-task,** at a natural break when the task is not finished yet, e.g. after a long investigation, before implementing.
- **Prefer `/clear` plus a new session** when the task is done.
- **Auto-compaction is a safety net, not a strategy.**

## 4. Models and effort

Pick both at the **start** of a session. Switching mid-session invalidates the cache.

| Work | Model | Effort |
|---|---|---|
| Everyday coding, bug fixes, docs, small features | **Sonnet** (`/model sonnet`) | medium (default); **high** for hard bugs |
| Planning a big phase, architecture, scientific-correctness reasoning, gnarly debugging | **Opus** | medium–high |
| Test runs, log reading, simple searches (via subagents) | **Haiku** | low–medium |

- Avoid `max` effort. The docs call it prone to overthinking, and thinking tokens bill as output.
- Check your consumption with `/usage`, and see what is filling the context with `/context`.

## 5. Planning big, phased work

This is the pattern used for v3.2–4.0, made cheaper.

1. **Plan session (Opus, plan mode).**
   - Say: "Read docs/ROADMAP.md; plan <goal>; interview me with questions."
   - Approve the plan. Have Claude save it to `docs/plans/<name>.md`, split into **waves of independent workstreams**. Each workstream lists files owned, tasks and acceptance tests.
   - Add the phase to `docs/ROADMAP.md`.
2. **Execution.** Use one of two options:
   - **Sequential (simplest, cheapest):** a new conversation per workstream: "Implement workstream W-2 from docs/plans/<name>.md on branch feat/x. Commit and push after each task."
   - **Parallel:** one small orchestrator conversation launches `workstream` subagents, one per branch in its own worktree, gives each the path to its brief file, and merges their branches. Keep the orchestrator lean: briefs live in files, not in chat; review screenshots and diffs, not full logs.
3. **Between waves:** run `/handoff`. If the orchestrator conversation has grown large, start a new one with the handoff prompt.
4. **Release:** run `/release x.y.z`.

> Parallel subagents finish sooner but consume your usage limit faster; they all draw on the same limits. With a 5-hour window, run at most 2–3 at a time. Check `/usage` before launching a wave.

## 6. When to use subagents

- **Yes:**
  - anything that floods context: running tests (`/test`), reading big logs, broad code searches (the Explore agent), UI screenshot review (`/visual-qa`)
  - independent parallel workstreams (`workstream`)
- **No:** iterative back-and-forth with you, or small edits. Do those in the main conversation.

## 7. Keeping docs current, so a new conversation knows everything

- **`docs/ROADMAP.md`** holds status, next step, follow-ups and decisions. It is updated by `/handoff` and at the end of every task (this is in CLAUDE.md's definition of done).
- **`CHANGELOG.md`** has one section per release, written by `/release`.
- **`OUTPUTS.md`** must change in the same commit as any new or changed output file or column.
- **`CLAUDE.md`** holds durable rules and commands only. When Claude makes the same mistake twice, add a one-line rule. Prune now and then; `/doctor` can suggest trims.
- **Priorities and decisions you state in chat:** tell Claude "add this to the ROADMAP decisions" so they survive the conversation.

## 8. Optional settings (not enabled; your call)

Shared settings go in `.claude/settings.json` (committed); personal ones in `.claude/settings.local.json` (git-ignored).

**Auto-lint after edits** (a hook):
```json
{ "hooks": { "PostToolUse": [ { "matcher": "Edit|Write",
  "hooks": [ { "type": "command",
    "command": ".build-venv/Scripts/python.exe -m ruff check --fix --quiet \"$CLAUDE_PROJECT_DIR\"" } ] } ] } }
```

**Fewer permission prompts** for safe commands:
```json
{ "permissions": {
  "allow": ["Bash(.build-venv/Scripts/python.exe -B -m unittest *)", "Bash(.build-venv/Scripts/python.exe -m ruff *)",
            "Bash(git status *)", "Bash(git diff *)", "Bash(git log *)"],
  "ask":   ["Bash(git push *)"] } }
```

## 9. Housekeeping

- **Old branches and worktrees.** Merged `fix/*`, `feat/*` and `chore/*` branches and `.claude/worktrees/*` can be deleted; see the ROADMAP follow-ups. Clean worktrees are removed automatically when a worktree session exits.
- **Before trusting a release's numbers,** do the manual GUI pass and, eventually, the criterion-validation run in `VALIDATION_PROTOCOL.md`.
