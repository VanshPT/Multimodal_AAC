# Multimodal AAC Chatbot (CSE 635 Project 4)

Training-free, retrieval-based AAC assistant built on Django with:
- partner chat mode (`Normal mode`)
- proactive suggestion mode (`Speak mode`)
- memory update + acknowledgement flow
- multimodal face/gesture conditioning (smile, confused, nod/shake simulation)

---

## 1) Environment Setup

### Prerequisites
- Python 3.10+
- `pip`
- Webcam access (optional, can use simulated faces)

### Install
```bash
pip install -r requirements.txt
python manage.py migrate
python scripts/generate_synthetic_user.py
```

### Optional LLM setup (`.env` in repo root)
```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Notes:
- `GEMINI_API_KEY` is preferred.
- `GOOGLE_API_KEY` is also supported as alias.

### Start app
```bash
python manage.py runserver
```
Open: [http://127.0.0.1:8000/aac/](http://127.0.0.1:8000/aac/)

---

## 2) Tester UI Settings (important)

Use these exact settings for consistent testing:

1. Click `START`
2. `User` -> `demo_user`
3. `Partner name` -> `Omer`
4. Turn ON:
   - `Memory Update`
   - `PB Enabled`
5. For expression tests:
   - turn ON `Sim Face`
   - select one preset from:
     - `smile`
     - `confused`
     - `nod_yes`
     - `shake_no`
     - `neutral`
6. Watch `Local time` chip in UI top bar (clock is live and sent to pipeline context).

---

## 3) General System Test Queries

### A) Core memory-grounding checks
Send each in `Normal mode`:

1. `Are we still on for the movie tonight at 7?`
2. `Can we meet before 6:30 at the campus bus stop?`
3. `Did you finish your CSE 635 slides for the 2:00 check-in?`
4. `Do you want a reminder for evening medication at 8:30?`
5. `Are we still doing cricket at 1 PM on Sunday?`

Expected:
- Replies should include concrete details from memory (time/place/plan), not only generic yes/no.

### B) Binary polarity test with face gestures
Use the same query multiple times while changing preset:

Query:
- `Do you want a prescription reminder tonight?`

Run with:
1. `nod_yes`
2. `shake_no`
3. `confused`

Expected:
- Always 3 options with variety.
- `nod_yes`: agree-like option appears first.
- `shake_no`: decline-like option appears first.
- `confused`: clarification-first tone.

### C) Tone-only differential test
Use the same query:
- `Are we still on for the movie tonight?`

Run with:
1. `smile`
2. `confused`

Expected:
- Both are memory grounded.
- Wording/tone differs (smile warmer, confused more cautious/clarifying).

---

## 4) Memory Update + Recall Test

Keep `Memory Update` ON.

### Insert new partner memory
1. `Tomorrow we have project rehearsal at 5:15 PM.`
2. `Please remind me to carry my charger tomorrow morning.`
3. `Tonight let's leave from the campus bus stop at 6:20.`

Select/confirm any generated response each turn.

Expected:
- Final output box shows `Memory Ack: ...`
- Ack mentions target bucket (`today_plans`, `next_days_plans`, `reminders`, etc.)

### Verify retrieval of newly added memory
1. `What did I add for tomorrow at 5:15?`
2. `What reminder did I ask for tomorrow morning?`
3. `What is our pre-movie leave plan tonight?`

Expected:
- Newly added items should be reflected in generated options.

---

## 5) Face/Camera Notes

- Native `FaceDetector` API may be unavailable on some browsers.
- The app uses MediaPipe when available and falls back gracefully.
- If webcam behavior is unstable, use `Sim Face` presets for deterministic testing.

---

## 6) Running Automated Checks

### Unit tests
```bash
python manage.py test home.aac.tests
```

### Evaluation script
```bash
python scripts/run_evaluation.py
```

Generated artifacts:
- `outputs/metrics_summary.json`
- `outputs/run_logs.jsonl`
- `outputs/sample_transcripts.md`

---

## 7) Quick Troubleshooting

- No responses shown:
  - ensure server is running
  - click `START` before sending partner message
- Face not changing:
  - use `Sim Face` mode first
  - check `faceStatus` text under video
- No memory acknowledgements:
  - ensure `Memory Update` checkbox is ON before confirming
- LLM not active:
  - check `.env` key
  - visit `/health/llm`
