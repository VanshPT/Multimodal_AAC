# Multimodal AAC Chatbot (CSE 635 Project 4 — Final)

A training-free, retrieval-grounded AAC (Augmentative and Alternative Communication)
chatbot that produces three identity-faithful response options per partner turn under
a 5-second SLA, conditioned on face, gesture, gaze, and biosignal channels.

**Authors**: Syed Omer Shah, Vansh Thakkar
**Course**: CSE 635 — NLP and Text Mining (Spring 2026)
**University at Buffalo, Department of Computer Science and Engineering**

---

## Submission Deliverables

| File | Purpose |
|------|---------|
| `Final_Report.pdf` | ACL-style technical paper (5 pages + references) |
| `Final_Report.tex` | LaTeX source for the report |
| `Final_Slides.pptx` | 10-slide in-class presentation deck |
| `nlp_video.mp4` | Demo recording (live UI walkthrough) |
| `outputs/eval_summary.json` | Aggregate evaluation metrics (n=20) |
| `outputs/eval_rows.json` | Per-case evaluation rows |
| `outputs/figures/` | All plots referenced by the report and slides |

---

## System Overview

A 7-node LangGraph-style pipeline (FaceCue → MultimodalMapping → ParallelPrep →
SourcePriorityPlanner → BucketSelector → Retrieve&Rerank → EvidenceRefiner →
GroundednessGuard → CandidateGenerator) with a deadline-raced LLM rewriter on top of
a rule-based floor. The 12-bucket memory schema covers family, social, scheduling,
plans, medical, work, food, routine, and four polarity-shaped reply pools.

### Five P4 Bonus Features (all implemented)

1. **Gaze-Based Retrieval Activation** — gazed bucket receives a +0.30 score boost
2. **Vocal vs Air-Sign Conflict Resolution** — deliberate spatial channel wins
3. **Acceptance-Weighted Bucket Priors** — bucket selection feeds back into ranking
4. **Latency-Optimised Fallback** — flash-lite → flash → rule-based floor inside 5 s
5. **Online Index Update on Selection** — selected phrase boosted +0.15 in the bank

Implementation: [home/aac/multimodal.py](home/aac/multimodal.py),
[home/aac/parallel_prep.py](home/aac/parallel_prep.py),
[home/aac/pipelines/nodes.py](home/aac/pipelines/nodes.py),
[home/aac/pipelines/normal_pipeline.py](home/aac/pipelines/normal_pipeline.py),
[home/aac/llm.py](home/aac/llm.py),
[home/aac/service.py](home/aac/service.py).

---

## Final Evaluation (n = 20 held-out cases)

| Metric | Value |
|---|---|
| BLEU-2 (peak) | 0.219 |
| ROUGE-L (peak) | 0.366 |
| Groundedness (peak) | 0.545 |
| NLI faithfulness | 0.80 |
| Intent accuracy | 0.65 |
| Bucket routing accuracy | 0.70 |
| Polarity adherence | 0.95 |
| Memory-write accuracy | 0.90 |
| Multimodal alignment | 0.875 |
| Latency avg / p95 | 1825 / 2428 ms |
| 5-second SLA | met on every case |
| LLM fallback engaged | 0.45 |

---

## 1) Environment Setup

### Prerequisites
- Python 3.10+
- `pip`
- Webcam (optional — `Sim Face` presets work without one)
- Optional: Gemini API key for the LLM rewriter

### Install
```bash
pip install -r requirements.txt
python manage.py migrate
python scripts/generate_synthetic_user.py
```

### LLM configuration (`.env` in repo root)
```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
```
`GOOGLE_API_KEY` is also accepted as an alias.

### Start the app
```bash
python manage.py runserver
```
Open: [http://127.0.0.1:8000/aac/](http://127.0.0.1:8000/aac/)

---

## 2) Tester UI Walkthrough

1. Click `START`.
2. `User` → `demo_user`.
3. `Partner name` → e.g. `Omer`.
4. Toggle ON: `Memory Update`, `PB Enabled`.
5. For deterministic emotion testing, enable `Sim Face` and pick one of:
   `smile`, `confused`, `nod_yes`, `shake_no`, `neutral`.
6. The `Local time` chip in the top bar is sent into pipeline context.

---

## 3) Test Queries

### A) Memory grounding
1. `Are we still on for the movie tonight at 7?`
2. `Can we meet before 6:30 at the campus bus stop?`
3. `Did you finish your CSE 635 slides for the 2:00 check-in?`
4. `Do you want a reminder for evening medication at 8:30?`
5. `Are we still doing cricket at 1 PM on Sunday?`

Expected: replies cite concrete details from memory, not generic yes/no.

### B) Polarity differential (gesture-driven)
Query: `Do you want a prescription reminder tonight?`
Run with `nod_yes`, `shake_no`, `confused` in turn.

Expected: agree-leading, decline-leading, and clarify-leading options respectively.

### C) Tone differential (face-driven)
Query: `Are we still on for the movie tonight?`
Run with `smile` then `confused`. Both grounded; tone differs.

---

## 4) Memory Update + Recall

Keep `Memory Update` ON, then add:
1. `Tomorrow we have project rehearsal at 5:15 PM.`
2. `Please remind me to carry my charger tomorrow morning.`
3. `Tonight let's leave from the campus bus stop at 6:20.`

Confirm a response each turn. The output box shows `Memory Ack: …` with the target
bucket. Then verify recall with:
1. `What did I add for tomorrow at 5:15?`
2. `What reminder did I ask for tomorrow morning?`
3. `What is our pre-movie leave plan tonight?`

---

## 5) Face / Camera Notes
- Native `FaceDetector` is unavailable on some browsers.
- The app uses MediaPipe + face-api.js when available and degrades gracefully.
- For deterministic runs, use `Sim Face` presets.

---

## 6) Reproducing the Evaluation

```bash
python scripts/run_evaluation.py
python scripts/build_extra_plots.py
```

Generated artifacts:
- `outputs/eval_summary.json`, `outputs/eval_rows.json`
- `outputs/figures/*.png`

To rebuild the report PDF and slides:
```bash
tectonic Final_Report.tex
python scripts/build_final_slides.py
```

---

## 7) Troubleshooting
- **No responses**: ensure server is running and `START` was clicked first.
- **Face state stuck**: switch to `Sim Face` and pick a preset.
- **No memory acks**: `Memory Update` must be ON before confirming.
- **LLM idle**: check `.env` and visit `/health/llm`.

---

## License
Course project — see [LICENSE](LICENSE).
