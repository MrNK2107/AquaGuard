# AquaGuard demo script (about 6 minutes)

**Audience:** an NGO programme lead, an ROV operator, or a judge. **The story:** "From a dive video to a report you can defend, with a person in the loop."

## Before you start (5 minutes, once)

```bash
pip install -r requirements.txt
# Optional: start with auth on, to show the key flow
export AQUAGUARD_API_KEYS="demo-key"
python -m uvicorn app.main:app --port 8000
```

- Have **one short dive clip** ready (10 to 60 s is plenty) and **a folder of 10 to 20 seabed stills**. The repository doesn't include footage. Any SeaClear test-split frames work, or footage of your own.
- On CPU each sampled frame takes about 0.35 to 0.45 s. A 60 s clip sampled every 2 s is ready in about 15 s.
- Optional pre-run, so you aren't waiting live: `python scripts/mission_report.py clip.mp4 --name "Harbour wall T3" --every 1 --pdf`.

## 1. The problem (30 s)

> "A one-hour cleanup dive gives you an hour of video. Today someone watches it and tallies tyres and nets in a spreadsheet. It takes a day, two people get different numbers, and the report doesn't say how it was counted."

Open `http://localhost:8000/`. The landing page states the claim and the limits in one screen.

## 2. Start a mission (60 s)

1. Click **Open console**. If auth is on, paste `demo-key` into the API key box. The key is stored for this browser tab only.
2. Click **New survey**. Upload the clip, name it ("Harbour wall T3"), and set the site. Leave *Auto-count above* at 0.50 and *sample every* at 1 to 2 s.
3. Click **Start survey**. The mission page shows progress while frames are detected in batches of 8.

> "Each sampled frame is stored at 1280 px. A reviewer will see exactly what the model saw."

## 3. Read the report (90 s)

On the mission page, scroll the embedded report:

- **Estimated items** for a video is a **range**, such as `3–41`. Explain: "The ROV passes the same tyre for ten seconds. We don't track objects yet, so we give the lower bound (the most of one class seen in a single frame) and the upper bound (every counted box)."
- **Inventory by class**. The hatched orange bars are boxes **not counted yet**, because they are below the threshold.
- **Timeline**. This shows where in the dive the debris is. Hover a bar to see which classes are in that frame.
- **Debris hotspots**. These are the busiest frames, spread across the dive, with boxes drawn.
- **How these numbers were made**. The thresholds, the counting rule and the benchmark caveat are printed in the report itself.

## 4. Review (90 s), the key moment

Click **Review detections**.

- The queue starts with the **most ambiguous** box, the one nearest the threshold. The focused box is outlined in orange on the frame.
- Press <kbd>Y</kbd> to confirm one box and <kbd>N</kbd> to mark one "not debris". Press <kbd>3</kbd> to relabel a box as a glass bottle and <kbd>U</kbd> to undo.
- Press <kbd>A</kbd> and drag a box around something the model missed, then pick its class.
- The progress bar and "*n* of *m* boxes human-reviewed" update as you go.

> "The model never has the last word. Every decision is stored next to the model's original answer."

Go back to the mission. The summary now shows the reviewed share, and the report says "*n*% human-reviewed".

## 5. Hand-offs (45 s)

- **Download PDF** gives the funder or municipality a report.
- **Detections CSV** is the analyst's spreadsheet: every box, its status and whether it was counted.
- **Corrections for retraining (.zip)** contains `corrections.jsonl`, plus `labels/`, `images/` and `data.yaml` in YOLO format, for fully reviewed frames only.

> "The weakest classes, glass bottles and cans, are where reviewers spend their time. Those corrections are the next training set, collected at the site where the model will be used."

## 6. It's an API too (30 s)

```bash
curl -s -X POST localhost:8000/api/v1/missions -H "X-API-Key: demo-key" \
  -F files=@clip.mp4 -F name="Scripted dive" -F every_s=2
curl -s localhost:8000/api/v1/missions -H "X-API-Key: demo-key"
curl -s -i localhost:8000/api/v1/missions                    # 401 without a key
```

Point out `X-RateLimit-Remaining` in the headers, and `Retry-After` on a 429.

## Questions you will get

- **"How accurate is it?"** It scores 92.32% mAP@50 on the SeaClear hold-out split. That split is per image, so expect lower accuracy at a new site. This is why review is part of the workflow.
- **"Does it work without internet?"** Yes. It runs on CPU, a mission is a plain folder, and `scripts/mission_report.py` needs no server.
- **"Can it count rope / fishing line / construction debris?"** Not yet. It knows six classes, and the report says that other litter is not counted.
- **"GPS?"** Not yet. Missions have a free-text site, and the timeline uses video time.

## If something goes wrong

- **401 in the console:** paste the key into the header box.
- **"Mission is processing" for a long time:** the CPU is busy, or `max_frames` is high. Sampling every 2 to 5 s keeps demos quick.
- **No frames:** the video codec may not be readable by OpenCV. Re-encode with `ffmpeg -i in.mov -c:v libx264 out.mp4`.
