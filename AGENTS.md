# Prompt Logging Protocol

## General Workflow
1. Track each batch of edits triggered by a user prompt by creating a fresh Markdown file under `.agent/changes/`. Use descriptive filenames such as `.agent/changes/2024-06-10_14-23-45.simple_change.md` so entries remain chronological and searchable.
2. Inside that file, paste every prompt that directly influenced the edits and record a concise summary of the resulting changes.
3. Repeat this process for each distinct batch of edits. Do not append unrelated changes to an existing log file.

## Recording Prompts After a Change
For every change-inducing prompt set:
- Open (or create) the relevant `.agent/changes/<name>.md` file.
- Append the literal prompt text in the order received, using the numbered block format below.
- Follow each prompt with a `Changes` section containing one or more bullet points summarizing the modifications triggered by that prompt.

Format for each prompt entry:
```
[Prompt <n>]
<verbatim prompt text>

[Changes]:
* <bullet describing the resulting edits>
* <another bullet describing the resulting edits>
* ...
```

## Preparing a Commit Message
When the user requests that a commit be prepared:
1. Read every Markdown file inside `.agent/changes/` and order them by filename to preserve chronology.
2. Combine their contents into a single message using the template:
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
5. Echo a command that the user can paste directly, chaining the commit with moving the directory into `.agent/done/` using the same name. Format:
   ```
   git commit -F .agent/commits/<YYYY-MM-DD_HH-MM-SS>.<summary>/commit.md && mv .agent/commits/<dirname> .agent/done/<dirname>
   ```

## Additional Notes
- Always capture prompts verbatim; do not paraphrase user requests in the `Prompt` blocks.
- Keep change summaries short and action-oriented.
- Include prompts that clarified requirements or triggered follow-up adjustments if those instructions influenced the final code.
- When a changes file covers work that was planned across earlier prompts, include those prior prompts as well—provided they are relevant to the current edits and have not already been logged elsewhere.
