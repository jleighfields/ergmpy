# Lessons

Corrections that changed how work is done here. Each names the mistake, why it
happened, and the rule that prevents it. Review this when starting related work.

## Do not cut an instruction because a capable reader would not need it

**What happened.** A review of `CLAUDE.md` and `.claude/` recommended deleting
roughly three quarters of both files, with most deletions justified as "generic
review practice any competent agent already applies unprompted."

**Why it is wrong.** The explicit instruction is what makes behavior consistent
across different models, different sessions, and agents of varying capability.
An instruction that is redundant for a strong reader is load-bearing for a
weaker one, and the failure when it is missing is silent and intermittent —
the kind that does not show up when you check.

**The rule.** Remove text only when it is *wrong* for this repo, *duplicated*
elsewhere, *misplaced* (it belongs in the README or a docstring), or
*contradicted* by another instruction. Verbosity on its own is not a defect.

## Fix the environment before deleting the instructions that use it

**What happened.** Every mechanical command in the seven `.claude/` files
failed — `ruff`, `pytest` and `detect-secrets` all reported `Failed to spawn`,
because the repo had no `pyproject.toml`. A review proposed deleting each
instruction that invoked them.

**Why it is wrong.** The instructions were right and the environment was
missing. Adding the project file made most of the proposed deletions
unnecessary, and it fixed two stale skill descriptions as a side effect.

**The rule.** When an instruction cannot be followed, ask first whether the
instruction or the environment is at fault. Deleting the caller is the right
answer only when the thing it calls should not exist.

## A skill that cannot establish its baseline must say so

**What happened.** The skills instructed an agent to run the mechanical passes
first and then read. With the tools absent, those commands errored, and nothing
told the agent what to do about it — so it would proceed to the reading pass
having verified nothing and report as though it had.

**Why it matters.** That is the silent no-op this repo's own rules warn about,
committed inside the files doing the warning. A check that cannot run is not a
check that passed.

**The rule.** Any instruction to run a command carries an instruction for what
to do when it fails to run. Report the gap; never continue as if it succeeded.
