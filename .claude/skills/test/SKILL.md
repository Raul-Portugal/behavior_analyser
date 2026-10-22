---
name: test
description: Run tests (named modules, or the full suite), ruff and the startup check via the test-runner subagent and report only failures.
argument-hint: "[test module(s), e.g. tests.test_workspace_trials_stage]"
---
Use the `test-runner` subagent to run these checks: $ARGUMENTS (if empty, the full suite).

Then summarize its report in at most 10 lines. If anything failed, propose the fix, but do not apply it unless I ask.
