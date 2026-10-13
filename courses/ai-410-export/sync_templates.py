#!/usr/bin/env python3
"""
Publish the AI 410 starter code as four GitHub template repos, one per module.

Each module repo holds that module's HOS starter and its Application Challenge
starter, each with its own Codespaces setup:

    ai410-module-N/
      README.md
      hos/                          the HOS N starter
      challenge/                    the Application Challenge N starter
      .devcontainer/hos/devcontainer.json
      .devcontainer/challenge/devcontainer.json
      .gitignore

The drafts in this repo stay the source of truth. Edit starter code here, then
sync. Students get whatever the template holds when they accept the module's
Classroom invite, so sync (and have reviewers re-test) before a module opens.

    # 1. Preview what would happen (default; changes nothing anywhere)
    python3 courses/ai-410-export/sync_templates.py --org cityuseattle

    # 2. Build the repo folders locally, e.g. to inspect or test them
    python3 courses/ai-410-export/sync_templates.py --build-only /tmp/ai410-templates

    # 3. Create or update the template repos on GitHub
    python3 courses/ai-410-export/sync_templates.py --org cityuseattle --apply
    python3 courses/ai-410-export/sync_templates.py --org cityuseattle --apply --module 3

Safety
  - Dry run unless --apply.
  - Copies only files git tracks (no venv/, .env, caches).
  - Refuses if the starter folders have uncommitted changes, so every sync
    matches a commit here (override with --allow-dirty).
  - Never force-pushes. Each sync is a normal commit on top of the repo's
    history; it warns if someone committed to the repo directly.
  - New repos are created private by default (--visibility to change).

Needs: git, and the GitHub CLI (`gh`) logged in as someone who can create repos
in the org.
"""

import argparse
import json
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STARTERS = "courses/ai-410-guides/starter-code"
MODULES = {
    1: ("Foundations & the Agentic Loop", "1–2", "hos-01-foundations-agentic-loop", "ac-01-clinic-booking"),
    2: ("Embeddings, Chunking & RAG", "3–4", "hos-02-embeddings-rag", "ac-02-helpdesk-retrieval"),
    3: ("Guardrails & Async Processing", "5–6", "hos-03-guardrails-async", "ac-03-refund-guardrail"),
    4: ("Mobile, Push & Observability", "7–8", "hos-04-mobile-observability", "ac-04-substitution-observability"),
}
SYNC_PREFIX = "Sync from stc-transformation"
DRAFT_SITE = re.compile(r"https://clarkngo\.github\.io/stc-transformation/(courses/[\w./-]+\.html)")


def run(cmd, cwd=None, check=True, capture=True):
    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=capture)
    if check and r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stderr.strip()}")
    return r


def tracked_files(folder):
    out = run(["git", "ls-files", "-z", "--", folder], cwd=REPO).stdout
    return [f for f in out.split("\0") if f]


def published_path(draft_page):
    """Where a draft page lands on the deployed site (same mapping as export_ai410.py)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from export_ai410 import PAGES
    return PAGES.get(draft_page)


def module_devcontainer(src_json, folder, label):
    cfg = json.loads(src_json)
    cfg["name"] = label
    cmd = cfg.get("postCreateCommand")
    if isinstance(cmd, list):
        cmd = " ".join(cmd)
    if cmd:
        cfg["postCreateCommand"] = f'cd "$(git rev-parse --show-toplevel)/{folder}" && {cmd}'
    cfg["workspaceFolder"] = "/workspaces/${localWorkspaceFolderBasename}/" + folder
    return json.dumps(cfg, indent=4, ensure_ascii=False) + "\n"


def build(n, dest, site_url, notes):
    title, weeks, hos, ac = MODULES[n]
    dest.mkdir(parents=True)
    ignores = []
    for folder, sub in (("hos", hos), ("challenge", ac)):
        root = f"{STARTERS}/{sub}"
        files = tracked_files(root)
        if not files:
            sys.exit(f"no tracked files in {root}")
        for f in files:
            rel = f[len(root) + 1:]
            if rel == ".devcontainer/devcontainer.json":
                label = f"Module {n} · " + ("HOS" if folder == "hos" else "Application Challenge")
                text = module_devcontainer((REPO / f).read_text(), folder, label)
                out = dest / ".devcontainer" / folder / "devcontainer.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text)
                continue
            if rel == ".gitignore":
                ignores += (REPO / f).read_text().splitlines()
            out = dest / folder / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            data = (REPO / f).read_bytes()
            if out.suffix in (".md", ".html", ".txt") and site_url:
                text = data.decode("utf-8")
                def relink(m):
                    pub = published_path(m.group(1))
                    if pub:
                        return site_url.rstrip("/") + "/" + pub
                    notes.append(f"module {n}: {folder}/{rel} links to a draft page that isn't published: {m.group(1)}")
                    return m.group(0)
                data = DRAFT_SITE.sub(relink, text).encode("utf-8")
            elif out.suffix in (".md", ".html", ".txt") and DRAFT_SITE.search(data.decode("utf-8", "ignore")):
                notes.append(f"module {n}: {folder}/{rel} links to the drafting site (pass --site-url to point it at the deployed course)")
            out.write_bytes(data)
        if not (dest / ".devcontainer" / folder / "devcontainer.json").exists():
            notes.append(f"module {n}: {sub} has no .devcontainer/devcontainer.json")
    base_ignores = [".env", ".env.*", "!.env.example", "venv/", ".venv/", "__pycache__/", "*.pyc", ".pytest_cache/", "node_modules/", ".DS_Store"]
    seen, lines = set(), []
    for line in base_ignores + [l for l in ignores if l.strip() and not l.startswith("#")]:
        if line not in seen:
            seen.add(line); lines.append(line)
    (dest / ".gitignore").write_text("# Applies to both folders.\n" + "\n".join(lines) + "\n")
    guide = f"{site_url.rstrip('/')}/module-{n}/index.html" if site_url else None
    (dest / "README.md").write_text(f"""# AI 410 · Module {n}: {title}

Weeks {weeks}. This repo has both of this module's hands-on parts:

| Folder | What it is | Open it in Codespaces as |
|---|---|---|
| `hos/` | The HOS {n} starter code | **Module {n} · HOS** |
| `challenge/` | The Application Challenge {n} starter code | **Module {n} · Application Challenge** |

## Open it

On this repo's page, choose **Code → Codespaces → ⋯ → New with options**, pick
the configuration for the part you're working on, and create the codespace.
Each part sets up its own environment, so pick the one you need.

Each folder's own `README.md` says what's there and what to do.
{f"{chr(10)}The lesson and guides for this module: {guide}{chr(10)}" if guide else ""}
Never commit a `.env` file: your API keys stay in the codespace.
""")


def gh_repo(org, name):
    r = run(["gh", "repo", "view", f"{org}/{name}", "--json", "name,isTemplate,visibility,defaultBranchRef"], check=False)
    return json.loads(r.stdout) if r.returncode == 0 else None


def tree_diff(a, b):
    def files(root):
        return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts}
    fa, fb = files(a), files(b)
    added = sorted(set(fb) - set(fa)); removed = sorted(set(fa) - set(fb))
    changed = sorted(k for k in set(fa) & set(fb) if fa[k] != fb[k])
    return added, changed, removed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--org", default="cityuseattle", help="GitHub org that owns the template repos")
    ap.add_argument("--prefix", default="ai410-module-", help="repo name prefix (default ai410-module-)")
    ap.add_argument("--module", type=int, choices=sorted(MODULES), action="append", help="only these modules (repeatable)")
    ap.add_argument("--visibility", choices=["private", "internal", "public"], default="private", help="for newly created repos")
    ap.add_argument("--site-url", help="deployed course URL (e.g. https://…/ai-410) to point README links at")
    ap.add_argument("--build-only", metavar="DIR", help="write the repo folders to DIR and stop")
    ap.add_argument("--apply", action="store_true", help="actually create/update the repos on GitHub")
    ap.add_argument("--allow-dirty", action="store_true", help="sync even if the starter folders have uncommitted changes")
    a = ap.parse_args()
    mods = a.module or sorted(MODULES)

    dirty = run(["git", "status", "--porcelain", "--", STARTERS], cwd=REPO).stdout.strip()
    if dirty and not a.allow_dirty and not a.build_only:
        sys.exit("starter code has uncommitted changes; commit them first (or pass --allow-dirty):\n" + dirty)
    sha = run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO).stdout.strip()

    notes = []
    if a.build_only:
        out = Path(a.build_only).resolve()
        if REPO == out or REPO in out.parents:
            sys.exit("refusing to build inside the drafting repo")
        for n in mods:
            d = out / f"{a.prefix}{n}"
            if d.exists():
                shutil.rmtree(d)
            build(n, d, a.site_url, notes)
            print(f"built {d}")
        for x in notes:
            print("NOTE", x)
        return

    if run(["gh", "auth", "status"], check=False).returncode != 0:
        sys.exit("the GitHub CLI isn't logged in; run: gh auth login")
    for n in mods:
        name = f"{a.prefix}{n}"
        info = gh_repo(a.org, name)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); built = tmp / "built"; work = tmp / "repo"
            build(n, built, a.site_url, notes)
            if info is None:
                print(f"\n{a.org}/{name}: does not exist -> {'creating' if a.apply else 'would create'} ({a.visibility}, template)")
                n_files = sum(1 for p in built.rglob('*') if p.is_file())
                print(f"  {n_files} files")
                if not a.apply:
                    continue
                run(["gh", "repo", "create", f"{a.org}/{name}", f"--{a.visibility}",
                     "--description", f"AI 410 Module {n} ({MODULES[n][0]}): HOS and Application Challenge starter code"])
                work.mkdir()
                run(["git", "init", "-q", "-b", "main"], cwd=work)
                run(["git", "remote", "add", "origin", f"https://github.com/{a.org}/{name}.git"], cwd=work)
            else:
                run(["gh", "repo", "clone", f"{a.org}/{name}", str(work), "--", "-q"])
                last = run(["git", "log", "-1", "--format=%s"], cwd=work, check=False).stdout.strip()
                if last and not last.startswith(SYNC_PREFIX):
                    print(f"WARNING {a.org}/{name}: last commit wasn't made by this script ({last!r}); its changes will be replaced by the drafts")
            added, changed, removed = tree_diff(work, built)
            if info is not None:
                print(f"\n{a.org}/{name}: {len(added)} added, {len(changed)} changed, {len(removed)} removed"
                      + ("" if info.get("isTemplate") else "  (not marked as a template yet)"))
                for label, items in (("+", added), ("~", changed), ("-", removed)):
                    for f in items[:15]:
                        print(f"  {label} {f}")
                    if len(items) > 15:
                        print(f"  {label} … and {len(items) - 15} more")
            if not a.apply:
                continue
            if added or changed or removed:
                for p in work.iterdir():
                    if p.name != ".git":
                        shutil.rmtree(p) if p.is_dir() else p.unlink()
                for p in built.iterdir():
                    shutil.copytree(p, work / p.name) if p.is_dir() else shutil.copy2(p, work / p.name)
                run(["git", "add", "-A"], cwd=work)
                run(["git", "commit", "-q", "-m", f"{SYNC_PREFIX} {sha}"], cwd=work)
                run(["git", "push", "-q", "origin", "HEAD:main"], cwd=work)
                print("  pushed")
            else:
                print("  already up to date")
            run(["gh", "api", "-X", "PATCH", f"repos/{a.org}/{name}", "-F", "is_template=true"])
            print("  marked as a template")
    for x in notes:
        print("NOTE", x)
    if not a.apply:
        print("\nDry run: nothing was changed. Add --apply to create/update these repos.")


if __name__ == "__main__":
    main()
