# Prompt Logging Protocol

## General Workflow
1. Track each batch of edits triggered by a user prompt by creating a fresh Markdown file under `.agent/prepare/`. Use descriptive filenames such as `.agent/prepare/2024-06-10.simple_change.md` so entries remain chronological and searchable.
2. Inside that file, paste every prompt that directly influenced the edits and record a concise summary of the resulting changes.
3. Repeat this process for each distinct batch of edits. Do not append unrelated changes to an existing log file.

## Recording Prompts After a Change
For every change-inducing prompt set:
- Open (or create) the relevant `.agent/prepare/<name>.md` file.
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
1. Read every Markdown file inside `.agent/prepare/` and order them by filename to preserve chronology.
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
3. Materialize the commit message by writing it to `.agent/commits/<YYYY_MM_DD>.<summary>`. Create any missing directories along the way. If there is exactly one `.agent/prepare/` file involved, reuse that file's base name (e.g., `.agent/prepare/2024-06-10.simple_change.md` → `.agent/commits/2024-06-10.simple_change`).
4. Echo a command that the user can paste directly, chaining the commit with moving the generated file into `.agent/committed/` using the same filename. Format:
   ```
   git commit -F .agent/commits/<YYYY_MM_DD>.<summary> && mv .agent/commits/<filename> .agent/committed/<filename>
   ```
5. After the user confirms the commit has been created, archive the processed files inside `.agent/prepare/` by moving the file to `.agent/committed/`. Do not change the name when moving the file.
6. After archiving the prepare files, move the generated commit message from `.agent/commits/` to `.agent/committed/` with the exact same filename.

## Additional Notes
- Always capture prompts verbatim; do not paraphrase user requests in the `Prompt` blocks.
- Keep change summaries short and action-oriented.
- Include prompts that clarified requirements or triggered follow-up adjustments if those instructions influenced the final code.
