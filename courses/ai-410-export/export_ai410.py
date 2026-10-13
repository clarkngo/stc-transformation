#!/usr/bin/env python3
"""
Export the student-facing AI 410 course from the stc-transformation drafts
into a deployable folder (for example ../stc/ai-410).

    python3 courses/ai-410-export/export_ai410.py --out ../stc/ai-410
    python3 courses/ai-410-export/export_ai410.py --check     # verify only, write nothing

What it does
  1. Copies only the pages in PAGES, renamed into one folder per module.
  2. Rewrites every link between them, and copies the assets they use.
  3. Fails loudly (exit 1, with a list) if anything would leak or break:
       - a link to a page that isn't published (instructor pages, the
         schedule/design pages, the syllabus, legacy pages, ...)
       - a link or #anchor that doesn't resolve
       - instructor-only content on a published page (faculty notes,
         teaching notes, links to the teaching guide or instructor hub)
       - a referenced asset that doesn't exist
  Warnings (exit 0 unless --strict): starter-code links that still point
  at the drafting repo instead of the course's template repos.

No dependencies beyond the Python standard library.
"""

import argparse
import posixpath
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # stc-transformation/
SRC = "courses"                                     # paths below are relative to REPO

# draft page -> published path (inside the output folder)
PAGES = {
    "courses/ai-410-interactive/index.html": "index.html",
    "courses/ai-410-guides/account-setup-guide.html": "start-here/accounts.html",
    "courses/ai-410-guides/tools-you-will-use.html": "start-here/tools.html",
    "courses/ai-410-interactive/concept-debate-example.html": "reference/concept-debate-example.html",
    "courses/ai-410-guides/concept-companion.html": "reference/concept-companion.html",
    "courses/ai-410-guides/starter-code-walkthrough.html": "reference/starter-code-walkthrough.html",
    "courses/ai-410-interactive/team-project.html": "module-5/index.html",
}
_HOS = {1: "hos-01-foundations-agentic-loop", 2: "hos-02-embeddings-rag", 3: "hos-03-guardrails-async", 4: "hos-04-mobile-observability"}
_SIM = {1: "sim01-agentic-loop", 2: "sim02-rag-pipeline", 3: "sim03-guardrails-queue", 4: "sim04-push-observability"}
_AC = {1: "ac-01-clinic-booking", 2: "ac-02-helpdesk-retrieval", 3: "ac-03-refund-guardrail", 4: "ac-04-substitution-observability"}
for n in (1, 2, 3, 4):
    PAGES[f"courses/ai-410-interactive/hos-0{n}-lesson.html"] = f"module-{n}/index.html"
    PAGES[f"courses/ai-410-guides/{_HOS[n]}.html"] = f"module-{n}/hos-guide.html"
    PAGES[f"courses/ai-410-interactive/{_SIM[n]}.html"] = f"module-{n}/simulator.html"
    PAGES[f"courses/ai-410-guides/{_AC[n]}.html"] = f"module-{n}/application-challenge.html"

# Shared assets get fixed names; any other local file a page uses is copied under assets/files/.
ASSETS = {
    "courses/ai-410-assets/course.css": "assets/course.css",
    "courses/ai-410-assets/course.js": "assets/course.js",
}

# Text that must never appear on a published page.
FORBIDDEN = [
    (r'class="faculty"', "faculty notes block"),
    (r'>\s*Teaching note\s*<', "instructor teaching note"),
    (r'(teaching-guide|instructor|ac-exemplars|ac-faculty-notes)\.html', "link to an instructor page"),
    (r'AI-410-syllabus[\w-]*\.html', "link to the site syllabus (students get the docx in Brightspace)"),
    (r'bsai-course-design-constraints\.html', "link to internal design constraints"),
    (r'(?:href|src)="https?://(?:localhost|127\.0\.0\.1)', "link to a localhost URL"),
]
DRAFT_REPO_CODE = re.compile(r"github\.com/clarkngo/stc-transformation/tree/main/courses/ai-410-guides/(starter-code|answer-key)/")

ATTR = re.compile(r'(\b(?:href|src)=")([^"]+)(")')
SCRIPT_HTML = re.compile(r"""(['"])([\w./-]+\.html(?:#[\w-]*)?)\1""")
ID = re.compile(r'\bid="([^"]+)"')


def is_external(u):
    return bool(re.match(r"^(?:[a-z][a-z0-9+.-]*:|//)", u, re.I))


YT = re.compile(r"(?:youtube(?:-nocookie)?\.com/(?:watch\?v=|embed/)|youtu\.be/)([\w-]{11})")


def check_videos(sources):
    """A YouTube video that was removed or made private makes oEmbed return an error."""
    import json
    import urllib.error
    import urllib.request
    found = {}
    for src, text in sources.items():
        for vid in YT.findall(text):
            found.setdefault(vid, src)
    problems = []
    for vid, src in sorted(found.items()):
        url = "https://www.youtube.com/oembed?format=json&url=https://www.youtube.com/watch?v=" + vid
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                title = json.load(r).get("title", "")
            print(f"video ok: {vid} {title}")
        except urllib.error.HTTPError as e:
            problems.append(f"{src}: YouTube video {vid} is unavailable (HTTP {e.code})")
        except OSError as e:
            problems.append(f"{src}: couldn't reach YouTube to check video {vid} ({e})")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="output folder, e.g. ../stc/ai-410")
    ap.add_argument("--check", action="store_true", help="verify only; write nothing")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    ap.add_argument("--check-videos", action="store_true", help="also confirm every embedded/linked YouTube video still exists (needs network)")
    a = ap.parse_args()
    if not a.check and not a.out:
        ap.error("give --out FOLDER, or --check")

    errors, warnings = [], []
    sources = {}
    for src in PAGES:
        p = REPO / src
        if not p.exists():
            errors.append(f"missing page in drafts: {src}")
        else:
            sources[src] = p.read_text(encoding="utf-8")
    page_ids = {src: set(ID.findall(t)) for src, t in sources.items()}
    copies = dict(ASSETS)   # draft asset -> published path
    out_pages = {}

    for src, text in sources.items():
        dst = PAGES[src]
        src_dir = posixpath.dirname(src)
        dst_dir = posixpath.dirname(dst)

        for pat, why in FORBIDDEN:
            for m in re.finditer(pat, text):
                line = text.count("\n", 0, m.start()) + 1
                errors.append(f"{src}:{line}: {why}: {m.group(0)!r}")
        for m in DRAFT_REPO_CODE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            warnings.append(f"{src}:{line}: {m.group(1)} link points at the drafting repo (switch to the template repo / Classroom link)")

        def resolve(url, where):
            if url.startswith("#"):
                if url[1:] and url[1:] not in page_ids.get(src, ()):
                    errors.append(f"{src}: {where} {url!r}: no element with that id on this page")
                return url
            path, _, frag = url.partition("#")
            path = path.split("?")[0]
            target = posixpath.normpath(posixpath.join(src_dir, path))
            if target in PAGES:
                if frag and frag not in page_ids.get(target, ()):
                    errors.append(f"{src}: {where} {url!r}: {target} has no #{frag}")
                new = posixpath.relpath(PAGES[target], dst_dir or ".")
                return new + (("#" + frag) if frag else "")
            if target.endswith(".html"):
                errors.append(f"{src}: {where} {url!r} -> {target} is not a published page")
                return url
            if not (REPO / target).exists():
                errors.append(f"{src}: {where} {url!r} -> missing file {target}")
                return url
            if target not in copies:
                rel = target[len(SRC) + 1:] if target.startswith(SRC + "/") else target
                copies[target] = "assets/files/" + rel
            return posixpath.relpath(copies[target], dst_dir or ".") + (("#" + frag) if frag else "")

        def attr(m):
            url = m.group(2)
            if is_external(url) or url.startswith("data:"):
                return m.group(0)
            return m.group(1) + resolve(url, "link") + m.group(3)

        new = ATTR.sub(attr, text)

        def script_block(m):
            def lit(mm):
                return mm.group(1) + resolve(mm.group(2), "script string") + mm.group(1)
            return SCRIPT_HTML.sub(lit, m.group(0))
        new = re.sub(r"<script\b(?![^>]*\bsrc=)[^>]*>.*?</script>", script_block, new, flags=re.S)
        out_pages[dst] = new

    if a.check_videos:
        errors += check_videos(sources)

    for w in warnings:
        print("WARNING", w)
    if a.strict:
        errors += warnings
    if errors:
        print(f"\nEXPORT FAILED: {len(errors)} problem(s)", file=sys.stderr)
        for e in errors:
            print("  ERROR", e, file=sys.stderr)
        sys.exit(1)

    print(f"OK: {len(out_pages)} pages, {len(copies)} assets, {len(warnings)} warning(s)")
    if a.check:
        return
    out = Path(a.out).resolve()
    if out == REPO or REPO in out.parents:
        sys.exit("refusing to export inside the drafting repo")
    marker = out / ".ai410-export"
    if out.exists() and any(out.iterdir()) and not marker.exists():
        sys.exit(f"{out} is not empty and wasn't made by this script; pick an empty folder")
    if out.exists():
        shutil.rmtree(out)
    for dst, html in out_pages.items():
        p = out / dst
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")
    for src, dst in copies.items():
        p = out / dst
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / src, p)
    marker.write_text("Generated by stc-transformation/courses/ai-410-export/export_ai410.py. Edit the drafts, not these files.\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
