"""Shared prompt components for chat and agent systems.

Prompt parts are composable strings. Schema placeholders ({agent_schema})
are substituted at runtime by the caller.
"""

# ---------------------------------------------------------------------------
# Shared: Tool & function documentation (used by both chat and agent)
# ---------------------------------------------------------------------------

TOOL_DOC_READ_FUNCTIONS = """\
## Tool: run_python
You have one tool: `run_python`. It executes Python code in an isolated container.

**Always `print()` results.** If you call `db()` or any function without printing, \
you will not see the output. Every tool call should produce visible output.

### Python functions to retrieve data (no import needed; these are NOT SQL — do not use inside db())
- `db(sql, *params)` — read-only SQL (SELECT/WITH only). Supports `$1` params: `db("SELECT * FROM t WHERE id = $1", 123)`. Returns list[dict].
- `fmt(rows, max_rows=50, max_cell=80)` — format rows as compact table for inspection.
- `file_info(id)` — metadata for a conversation file: {filename, size_bytes, mime_type, nas_path, project_name, extracted_text}.
- `file_text(id_or_path)` — extract text from a file (PDF, docx, pptx, xlsx, csv, txt). Returns string.
- `file_image(id_or_path, page=None, max_dim=None)` — queue an image for you to see. For PDFs pass page number.
- `describe_image(id_or_path, question=None, page=None)` — send image to a vision model, get text description back.
- `download_file(content_hash)` — download a NAS file into /work/ by content_hash. Returns local path.
- `download_craft_file(storage_path)` — download a Craft doc media file into /work/. Returns local path.
- `download_url(file_id)` — get a download URL for a file the user can click.
- `web_search(query, depth='standard')` — search the web. Returns list of {name, url, content}. Use `depth='deep'` for thorough results.
- `fetch_url(url)` — fetch a webpage, returns content as markdown.

Example:
```python
tasks = db("SELECT * FROM v_project_tasks WHERE project_id = 635709 AND status = 'new'")
print(fmt(tasks))
```"""

TOOL_DOC_WRITE_FUNCTIONS = """\
### Python functions to write data (no import needed; these are NOT SQL — do not use inside db())
- `add_activity_entry(project_id, logged_at, category, summary, source_event_ids=[], kgr_codes=[], involved_persons=[])` — insert Tier 3 entry. Returns UUID.
- `update_activity_entry(entry_id, summary=None, category=None, kgr_codes=None, involved_persons=None, append_source_event_ids=[])` — amend a recent Tier 3 entry (< 48h old).
- `update_project_status(project_id, markdown)` — replace Tier 2 status. Rejected if new text < half current length.
- `update_project_profile(project_id, markdown)` — replace Tier 1 profile. Same length protection.

Examples:
```python
result = update_project_status(635709, "## Aktueller Stand\\nProjekt in Bauphase.")
print(result)

entry_id = add_activity_entry(
    project_id=635709,
    logged_at="2026-03-06T19:05:00+00:00",
    category="progress",
    summary="Kälteplanung für Serverraum gestartet.",
    source_event_ids=[1, 2],
    kgr_codes=["KGR 434"],
)
print(entry_id)
```"""

TOOL_DOC_ENVIRONMENT = """\
### Environment
- Full Python with all standard libraries. `import` works normally.
- `subprocess`, `os`, `pathlib`, `open()` all work.
- /work/ is your workspace. Files attached to the conversation are pre-populated there.
- New files saved to /work/ are automatically uploaded and available to the user.
- Variables persist across tool calls within this response."""

# ---------------------------------------------------------------------------
# Shared: Schema section
# ---------------------------------------------------------------------------

SCHEMA_SECTION = """\
## Database Schema
{agent_schema}

For the complete schema (all tables, columns, FKs), call: `db("SELECT get_full_schema()")`
For a specific schema: `db("SELECT get_full_schema('missive')")`"""

# ---------------------------------------------------------------------------
# Shared: File inspection guidance
# ---------------------------------------------------------------------------

FILE_INSPECTION = """\
## Inspecting files
- **Craft doc media**: URLs in Craft markdown like `.../craft-files/DOC_ID/BLOCK_ID_filename.pdf` — \
use `download_craft_file("DOC_ID/BLOCK_ID_filename.pdf")` to pull into /work/.
- **NAS files**: Use `v_project_files` to find files (has `content_hash`), then `download_file(content_hash)`.
- **Email attachments**: Use `v_project_emails` to find emails with attachments, then `v_project_files` \
(filter by `source_email_subject`) to find the downloaded file and its `content_hash`.

Once in /work/, use `file_image(path)` for images, `file_text(path)` for PDFs/docs, \
or `describe_image(path, question)` for vision analysis."""

# ---------------------------------------------------------------------------
# Shared: Link generation
# ---------------------------------------------------------------------------

LINK_GENERATION = """\
## Generating Links
When you mention specific tasks, emails, projects, or documents, always include clickable Markdown links.

- **Teamwork task**: `[Task Name](https://ibhelm.teamwork.com/#/tasks/{task_id})` — `v_project_tasks` has a `url` column
- **Teamwork project**: `[Project Name](https://ibhelm.teamwork.com/app/projects/{project_id})`
- **Missive conversation**: `[Subject](https://mail.missiveapp.com/#inbox/conversations/{conversation_id})` — `v_project_emails` has a `missive_url` column
- **Craft document**: `[Title](craftdocs://open?blockId={document_id})` — `v_project_craft_docs` has a `craft_url` column"""
