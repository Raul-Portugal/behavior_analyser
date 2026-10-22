---
name: handoff
description: Wrap up the current session so a fresh conversation can continue - update docs/ROADMAP.md, commit/push, and print a short continuation prompt.
---
Wrap up this session so that a brand-new conversation can continue without any of this chat:

1. **Commit and push.** Make sure all finished work is committed and pushed. Work in progress goes on its own branch.
2. **Update `docs/ROADMAP.md`.**
   - Under "Current status", record what is done, what is in progress and on which branch, and any open PRs.
   - Set "Next step" to concrete, ordered actions.
   - Add new follow-ups and remove completed ones.
   - Keep the file short.
3. **Update the other docs.** If behaviour, outputs or commands changed, update `CHANGELOG.md`, `OUTPUTS.md` or `CLAUDE.md`. Edit `CLAUDE.md` only for durable rules or commands.
4. **Commit and push** these documentation changes.
5. **Print a continuation prompt**, under 10 lines, that I can paste into a new conversation. Example: "Read docs/ROADMAP.md. Continue with <next step> on branch <x>. Plan first."
