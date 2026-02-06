# Prompt Logging Protocol

## General Workflow
1. Track each batch of edits triggered by a user request by creating a fresh Markdown file under `.agent/changes/`. Use descriptive filenames such as `.agent/changes/2024-06-10_14-23-45.simple_change.md` so entries remain chronological and searchable.
2. Inside that file, capture all relevant prior conversation that influenced the edits, including back-and-forth between user prompts and Codex responses, and record a concise summary of the resulting changes.
3. Repeat this process for each distinct batch of edits. Do not append unrelated changes to an existing log file.
4. After each change batch, prepare a full commit message and create an actual commit that includes only the relevant changes.

## Recording Conversation After a Change
For every change-inducing prompt set:
- Open (or create) the relevant `.agent/changes/<name>.md` file.
- Append the literal user prompts and the Codex responses that influenced the edits in strict chronological order, using the numbered block format below.
- Follow each prompt/response group with a `Changes` section containing one or more bullet points summarizing the modifications triggered by that prompt/response thread.

Format for each prompt entry:
```
[Prompt <n>]
<verbatim prompt text>

[Response <n>]
<verbatim Codex response text>

[Changes]:
* <bullet describing the resulting edits>
* <another bullet describing the resulting edits>
* ...
```

## Preparing a Commit Message and Committing
After each change batch:
1. Read the relevant Markdown file(s) inside `.agent/changes/` and order them by filename to preserve chronology.
   - If the change batch is scoped to a subset of files, only use the `.agent/changes/` entries relevant to that batch.
2. Combine their contents into a single commit message using the template:
   ```
   <one line summary>

   <summary body>

   [Prompt 1]
   <literal text of prompt 1>

   [Changes]:
   * <bullet point of changes from prompt 1>
   * <another bullet point of changes from prompt 1>
   * ...

   [Prompt 2]
   <literal text of prompt 2>

   [Changes]:
   * <bullet point of changes from prompt 2>
   * <another bullet point of changes from prompt 2>
   * ...
   ```
3. Materialize the commit message by creating a directory `.agent/commits/<YYYY-MM-DD_HH-MM-SS>.<summary>/`. Inside that directory, write the combined message to `commit.md`. If there is exactly one `.agent/changes/` file involved, reuse that file's base name for the directory (e.g., `.agent/changes/2024-06-10_14-23-45.simple_change.md` → `.agent/commits/2024-06-10_14-23-45.simple_change/`).
4. For every `.agent/changes/` file included in the commit, move it into a `changes/` subdirectory under the commit directory while preserving filenames (e.g., `.agent/changes/foo.md` → `.agent/commits/<dirname>/changes/foo.md`). Ensure the `changes/` directory is created first.
5. Stage only the relevant source changes for the commit (use `git add -p` when unrelated working changes exist). Do **not** stage `.agent/` files.
6. Create the commit using `git commit -F .agent/commits/<YYYY-MM-DD_HH-MM-SS>.<summary>/commit.md`.
7. Move the commit directory into `.agent/done/` using the same name (e.g., `.agent/commits/<dirname>` → `.agent/done/<dirname>`).

## Additional Notes
- Always capture prompts verbatim; do not paraphrase user requests in the `Prompt` blocks.
- Always capture the Codex responses verbatim in the `Response` blocks when they influenced the changes.
- Keep change summaries short and action-oriented.
- Include prompts that clarified requirements or triggered follow-up adjustments if those instructions influenced the final code.
- When a changes file covers work that was planned across earlier prompts, include those prior prompts as well—provided they are relevant to the current edits and have not already been logged elsewhere.
- If a prompt references earlier context (for example: "as discussed", "earlier recommendation", "earlier discussion", "that plan"), include the referenced prior prompt text verbatim in the same changes log and in the final commit message so the commit is self-contained.
- Treat terse implementation prompts (for example: "ok, do it", "implement now", "ship it", "apply it") as references to prior context; include the concrete earlier discussion points they rely on.
- Avoid ambiguous prompt-only commit sections; spell out the concrete recommendation/details directly in the commit message body or prompt blocks.
- Always use `uv` to run Python code.
- Keep `.agent/` as local bookkeeping only; do not add `.agent` files to git commits.
- For evaluation/experiment work (for example: hyperparameter sweeps, exploratory trainer edits, repeated run-analysis loops), always use a dedicated git worktree/branch rather than editing directly on `main`.
- When creating worktrees, name them using `<concept>.<YYYY-MM-DD_HH-MM-SS>`.
- Only modify `main` directly when explicitly doing collaborative in-place changes requested by the user.
- Prefer explicit assertions for invalid preconditions in diagnostics/utilities instead of returning empty placeholders; fail fast to surface misconfigured calls.
- If a function depends on a feature toggle or required module, assert with a clear error message rather than silently skipping.
- Use early assertions for minimum sequence length/shape requirements to avoid silently degraded metrics.

## jepa_world_model_trainer.py Changes
- When making non-trivial changes to `jepa_world_model_trainer.py`, run a very short training run up to the first visualization/planning dump (10 steps in the current config).
- Use a title indicating it is a test run (e.g., `test: <short description>`).
- For pipeflush smoke runs, use `data.gwbasic_rand_corner_loops2` unless explicitly overridden by the user.

## Available skills
- analysis: Analyze JEPA world model experiment behavior with a reproducible workflow for worktrees, hypotheses, utility scripts, and result logs. (file: `/Users/dave/rl/nes-ai/skills/analysis/SKILL.md`)
- jepa-planning-experiment: Run iterative JEPA world model trainer planning experiments focused on h-based planning quality. (file: `/Users/dave/rl/nes-ai/skills/jepa-planning-experiment/SKILL.md`)
- pipeflush: Run a short pipeflush smoke test for non-trivial changes to `jepa_world_model_trainer.py`, including a 10-step run to first visualization/planning dump. (file: `/Users/dave/rl/nes-ai/skills/pipeflush/SKILL.md`)
