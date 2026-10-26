---
name: release
description: Cut a release of Behavioral Maze Analyzer (version bump, changelog, PR, merge, tag, GitHub release).
argument-hint: "<version, e.g. 4.1.0>"
disable-model-invocation: true
---
Release version $ARGUMENTS. Stop and ask me before every step marked (confirm).

1. **Check the branch.** Make sure we are on `release/$ARGUMENTS` (or the branch I name). Run the full suite, ruff and `--startup-check` with the `test-runner` subagent. Everything must be green.
2. **Bump the version.** Set it to $ARGUMENTS in `core/version.py`, in `version_info.txt` (all strings and tuples) and in the `requirements-lock.txt` header.
3. **Write the changelog.** Add a `## $ARGUMENTS` section to `CHANGELOG.md`, verified against `git log --oneline origin/main..HEAD`.
   - Lead with whether analysis results changed compared with the last release.
   - Also update `OUTPUTS.md` and `quickstart.md` if needed.
   - Update `docs/ROADMAP.md`.
4. **Commit, push, and open a PR** to `main` with `"C:\Program Files\GitHub CLI\gh.exe" pr create`. Include a summary, a test plan and a manual GUI checklist.
5. (confirm) **Merge.** Once CI is green and I have done the manual GUI pass, merge with `gh pr merge <n> --merge`.
6. (confirm) **Publish the GitHub release.** Run `gh release create v$ARGUMENTS --target <merge sha>`, with notes taken from the CHANGELOG section. The `release.yml` workflow then builds the exe and attaches it with its SHA-256. Verify that the assets appear.
7. **Update `docs/ROADMAP.md`.** Mark the release as released and set the next step.
