---
name: visual-qa
description: Render workspace stages/widgets offscreen and review them for UI problems, via the ui-reviewer subagent.
argument-hint: "[page names, e.g. trials arena; empty = all]"
---
Use the `ui-reviewer` subagent to render and review these pages: $ARGUMENTS (if empty, all workspace pages). Render at 1366x768 and 1920x1080, light and dark.

Show me its top findings, ordered by priority, and say which ones you recommend fixing now.
