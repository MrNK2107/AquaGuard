# AquaGuard: product brief

**One line:** AquaGuard turns ROV dive footage into a debris inventory that a cleanup crew can check, sign off and hand to a funder, and every correction a person makes becomes training data for the next model.

## Who it is for

| Audience | What they do today | What they need from AquaGuard |
| :--- | :--- | :--- |
| **ROV and dive-team leads on cleanup projects** (harbour authorities, aquaculture operators, contractors) | Watch hours of footage after the dive, tally items in a spreadsheet, pick targets for the next dive | A per-dive inventory by class, with the frames where debris clusters, ready the same evening |
| **Marine conservation NGOs** that run seabed cleanups and report to donors or municipalities | Hand-count from photos or diver logs; reports differ from project to project | A consistent, dated report that shows its counting rules and how much of it a person has checked |
| **Researchers and survey analysts** | Label frames by hand in a general annotation tool to build their own dataset | A queue that shows the doubtful detections first, and an export in YOLO format |

The person who uses it most is the **deck operator or analyst** in the hours after a dive: on a laptop, often on a boat or in a site office, sometimes in bright light, with bad or no connectivity.

## The problem

1. **Counting debris on video is slow and doesn't repeat well.** One dive gives tens of minutes of footage. Different people count the same frames differently, and nobody writes down how the count was done.
2. **Raw detector output isn't a number you can report.** A model that draws boxes on every frame counts the same tyre 40 times as the ROV drifts past. It also shows low-confidence guesses next to sure ones with nothing to tell them apart.
3. **The model is weakest exactly where cleanups happen.** AquaGuard's detector scores 92.32% mAP@50 on the SeaClear hold-out split. That split is per image, so frames from the same dives appear in both training and test, and a new harbour with different water and a different camera will score lower. A product that hides this misleads the people who depend on it.

## What AquaGuard does (the three headline features)

1. **Mission reports.** Upload one dive video (sampled every N seconds) or a batch of stills. You get an inventory by class, a per-frame timeline, hotspot frames with boxes drawn, and a "how these numbers were made" section, as a self-contained HTML file or a PDF. Video counts are given as a range: the lower bound is the per-class peak in any single frame, the upper bound is every counted detection. Without object tracking a range is the honest answer, and the report says so.
2. **Review queue.** Detections below the auto-count threshold are not counted until a person confirms them. The queue puts the most ambiguous boxes first (closest to the threshold). The keyboard flow is Y to confirm, N for not debris, 1 to 6 to relabel, A to draw a missed item and F to accept the rest of a frame. Decisions never overwrite the model's output. The export (`corrections.zip`) contains a JSONL file that records what the model said and what the reviewer said, plus YOLO labels and images for the fully reviewed frames only.
3. **A secured API.** The same features run over REST (`/api/v1/missions/...`), next to the single-frame `/detect` and `/reason` endpoints. API keys come from `AQUAGUARD_API_KEYS` and are checked in constant time. Each key, or each IP when keys are off, gets a token-bucket rate limit (`AQUAGUARD_RATE_LIMIT`) and receives `X-RateLimit-*` and `Retry-After` headers. When auth is off, every response says `X-AquaGuard-Auth: disabled`, so nobody mistakes a dev server for a secured one.

There is also a single-frame **reasoning assistant**, which predates this product work. Ask "how many tyres?" or "where is the net?" and the answer comes from the boxes. It refuses questions the boxes can't answer (depth, weight, brand, low confidence).

## Positioning

AquaGuard sits between a general detection model and a person with a spreadsheet. It does not try to be a labelling platform or a general vision platform.

These alternatives exist and we can describe them honestly:

| Alternative | What it is good at | Where AquaGuard differs |
| :--- | :--- | :--- |
| **Manual review plus a spreadsheet** (the common practice) | No setup, full human judgement | AquaGuard does the first pass and leaves only the doubtful boxes for a person, and the counting rule is written into every report |
| **Diver-logged citizen-science surveys**, e.g. PADI AWARE's Dive Against Debris | Large volunteer networks with a shared reporting form | Those surveys are logged by divers by hand. AquaGuard works from ROV or camera footage and keeps frame-level evidence. The two can complement each other |
| **General annotation tools** such as CVAT or Label Studio | Flexible labelling for any task and any classes | They are built for producing datasets, not survey reports. AquaGuard's review queue exists to settle the counts for a report, and its export plugs back into training |
| **General vision platforms** such as Roboflow or Ultralytics HUB | Training and deploying custom detectors | They provide the model layer. AquaGuard is a finished workflow for one job (seabed debris surveys) with a model already tuned for it |

We don't claim anything about these products beyond what's above. We haven't benchmarked AquaGuard against any of them.

## Differentiators (things that are true today)

- **Counts you can defend.** Every report shows its thresholds, its counting rule, how much has been reviewed (for example "0% human-reviewed") and the caveat about the benchmark.
- **The human decision always wins, and it is kept.** Model output and review decisions are stored separately, so the report can be re-issued after review and the corrections export shows the model's answer next to the reviewer's.
- **A loop that improves the model where it is used.** The review export is in YOLO format with a `data.yaml`, so it can be merged into the next fine-tuning run.
- **Runs on a boat.** CPU-only works (about 0.35 to 0.44 s per frame on a 4-thread laptop CPU, measured). Missions are plain folders (`missions/<id>/mission.json` plus JPEG frames) that can be copied to a USB stick. There is no database server, and the offline CLI (`scripts/mission_report.py`) needs no server either.
- **Readable on deck.** The console and reports use a high-contrast light theme, and the six class colours are the Okabe-Ito colour-blind-safe set. They pass a palette validator for lightness, chroma and normal-vision separation. Three of the colours are below 3:1 contrast on white, so every chart also has direct labels and a table.

## Brand

- **Name:** AquaGuard. **Descriptor:** *Debris survey for ROV cleanup crews.*
- **Voice:** a careful field technician. Short sentences, plain numbers, and caveats stated next to the claim rather than in a footnote. Avoid words like "state-of-the-art" or "revolutionary".
- **Palette:** deep sea `#0E3A4F` (header band), ink `#0B1D2A` (text), paper `#F6F4EF` (page), line `#D9D4C7`, and one accent, signal orange `#FF5F1F`. Orange marks actions and "awaiting review" and appears nowhere else. On orange, text is ink, not white, for contrast. Class colours: tyre `#0072B2`, net `#CC79A7`, plastic bottle `#56B4E9`, glass bottle `#009E73`, metal can `#E69F00`, plastic bag `#D55E00`. They are defined once in `app/brand.py`.
- **Type:** IBM Plex Sans for UI and reports, IBM Plex Mono for ids, kickers and keys, with system-font fallbacks so reports render offline.
- **Mark:** a ring in signal orange around a wave line, like a sonar ping over the surface (inline SVG in the console and on the landing page).

## What it is not (yet)

- **No object tracking.** That is why video counts are ranges. Adding tracking (for example ByteTrack over the sampled frames) is the most valuable next model feature.
- **No geo-referencing.** Missions have a free-text site but no GPS or ROV telemetry. The timeline uses video time, not position.
- **Only six classes.** Other litter (rope offcuts, fishing gear other than nets, construction debris) is not counted, and the report says so.
- **One process, one mission at a time.** Processing runs as an in-process background task with a shared detector, and rate limits are per process. This fits a single vessel laptop or a small server. A multi-tenant service would need a job queue and a shared limiter (for example Redis).
- **Not yet field-validated.** All numbers come from the SeaClear dataset. No cleanup crew has used it on a live project yet.
