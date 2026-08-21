# stc-transformation

Static HTML documentation hub for STC (School of Technology and Computing, City University of Seattle). Every doc is a standalone, self-contained HTML file — no build step, no backend. `index.html` is the front door.

## Initiative Hubs

As of 2026-08-20 the site is organized around 7 dedicated **Initiative Hub** pages — each one a curated, growing index of the meeting notes, briefings, research, and plans for one active STC initiative. They're linked from the "Initiative Hubs" section near the top of `index.html`.

| Hub file | Covers |
|---|---|
| `blue-tech-maritime-hub.html` | Blue Tech / maritime AI, the practitioner-publisher content model |
| `frontline-spd-hub.html` | Frontline workers / CityU × Seattle PD AI partnership |
| `cityux-hub.html` | CityUX — the open-ecosystem funnel, Open edX platform. **Also covers work filed under this initiative's earlier names: AILI (AI Lifelong Institute) and Applied Intelligence Center (AIC).** |
| `digital-twins-hub.html` | Digital Twins / Physical AI lab |
| `bsai-hub.html` | Bachelor of Science in AI (degree program) |
| `teaching-framework-hub.html` | STC's pedagogy — assessment/grading design, student engagement, faculty tooling, faculty-development learnings |
| `corporate-partnerships-hub.html` | Employer-facing AI-enablement / contract-training partnerships |

### Convention: keep hubs current

**Whenever a new meeting-notes page, briefing, or research doc is created that belongs to one of these initiatives, update the relevant hub page(s) in the same piece of work — don't leave it to a later cleanup pass.**

1. Open the hub file(s) the new doc belongs to (a doc can belong to more than one — e.g. a president briefing that touches both Blue Tech and CityUX gets a row in both).
2. Add a `doc-row` entry inside the appropriate `<div class="group">` (or add a new group if it doesn't fit an existing one), following the existing markup pattern: title, one-line description, date.
3. Bump the hub's document count in its `meta-row` chip (`N documents`) to match.
4. If the new doc doesn't fit any of the 7 initiatives, it's fine to leave it un-hubbed — not everything needs to belong to one (it can still live in `index.html`'s "Recently added" strip or an audience-band card as before).

This is separate from — and in addition to — the existing convention of adding new docs to the "Recently added" strip in `index.html`.

### Hub page template

Hub pages share one lightweight visual system (see `blue-tech-maritime-hub.html` for the clearest example): a `doc-header` (eyebrow, title, subtitle, meta chips, optional "Also known as" box for naming history), an `overview` prose block, and one or more `group` sections containing `doc-row` link rows. Keep new hubs or new sections within existing hubs consistent with this pattern rather than introducing new visual styles per-hub.

## Homepage structure

`index.html` order: header → hero (`#hubs`/`#recent-added` anchor targets) → Initiative Hubs grid → live client-side search (filters hub cards, recent-items, audience-band cards, and archive cards by title/description) → "Recently added" strip → audience-band sections (Grant Research, Blue Tech, First Responders, Academic Affairs, Leadership & Strategy, Faculty, Students) → archived program history.

The audience-band sections predate the hub system and are kept as a complementary "browse by role" view — they are not being phased out, just supplemented.
