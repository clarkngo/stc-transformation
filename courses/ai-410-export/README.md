# AI 410 export

Builds the student-facing AI 410 course from the drafts in this repo into a
self-contained folder you can deploy (for example `../stc/ai-410`, or upload
to Brightspace as a zip).

```bash
python3 courses/ai-410-export/export_ai410.py --check            # verify only
python3 courses/ai-410-export/export_ai410.py --out ../stc/ai-410
```

## What gets published

| Draft | Published as |
|---|---|
| `ai-410-interactive/index.html` (course home) | `index.html` |
| `ai-410-interactive/hos-0N-lesson.html` | `module-N/index.html` |
| `ai-410-guides/hos-0N-*.html` | `module-N/hos-guide.html` |
| `ai-410-interactive/sim0N-*.html` | `module-N/simulator.html` |
| `ai-410-guides/ac-0N-*.html` | `module-N/application-challenge.html` |
| `ai-410-interactive/team-project.html` | `module-5/index.html` |
| `ai-410-guides/account-setup-guide.html` | `start-here/accounts.html` |
| concept companion, starter-code walkthrough, Concept Debate example | `reference/` |
| `ai-410-assets/course.css`, `course.js` and any image a page uses | `assets/` |

Instructor pages (the instructor hub, teaching guide, AC example answers, AC
faculty notes), the schedule and syllabus pages, and legacy pages are never
published.

## It fails on purpose

The script exits with an error and a list of problems if a published page:

- links to a page that isn't published
- has a link or `#anchor` that doesn't resolve
- contains instructor-only content (faculty notes, teaching notes, or links to
  instructor pages)
- uses an asset that doesn't exist

Fix the draft, not the output. The output folder is regenerated every time and
carries a `.ai410-export` marker; the script won't write into a non-empty folder
it didn't create.

`--check-videos` also confirms every YouTube video a page embeds or links still exists
(it needs network access; the PR check runs it).

Warnings (`--strict` turns them into errors): starter-code links that still
point at this drafting repo instead of the course's template repos.

## Adding a page

Add it to `PAGES` in `export_ai410.py`, then run `--check`.

# Template repos: `sync_templates.py`

Publishes the starter code as four GitHub template repos, one per module
(`cityuseattle/ai410-module-1` … `-4`). Each holds that module's HOS starter
(`hos/`) and Application Challenge starter (`challenge/`), each with its own
Codespaces configuration ("Module N · HOS" / "Module N · Application
Challenge"). Students get a copy through the module's GitHub Classroom invite.

```bash
python3 courses/ai-410-export/sync_templates.py --org cityuseattle                       # dry run: what would change
python3 courses/ai-410-export/sync_templates.py --build-only /tmp/ai410-templates          # build locally to inspect
python3 courses/ai-410-export/sync_templates.py --org cityuseattle --apply                # create/update all four
python3 courses/ai-410-export/sync_templates.py --org cityuseattle --apply --module 3     # just one module
```

Add `--site-url https://…/ai-410` so the starter READMEs link to the deployed
course instead of this drafting site.

**Weekly loop:** a module's template is reviewed the week before it opens.
Fix anything in the drafts here, commit, `--apply --module N`, have the
reviewer re-test, then publish that module's Classroom invite in Brightspace.
Students copy the template when they accept, so later fixes don't reach them.

It's a dry run unless `--apply`, copies only git-tracked files, refuses to run
with uncommitted starter-code changes, never force-pushes, and creates new
repos private by default.
