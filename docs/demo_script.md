# Multimodal AAC Chatbot — Live Demo Script

**Authors**: Syed Omer Shah, Vansh Thakkar
**Course**: CSE 635 — NLP and Text Mining (Spring 2026)
**Target runtime**: 8–10 minutes for in-class presentation, ~5 minutes for the recorded video.

This script walks through the 5 required nodes plus all 5 P4 bonuses on the live UI.
Every block is labelled with the exact thing to say and the exact action to take.

---

## 0) Pre-Flight (do this once before the demo starts)

```bash
# from repo root
source .venv/bin/activate            # any venv with django, google-genai, pillow
python manage.py migrate
python scripts/generate_synthetic_user.py    # creates demo_user STM/LTM/PB
python manage.py runserver 8000
```

Open in Chrome: `http://127.0.0.1:8000/aac/`

UI sanity check (top bar):
- Click **START**
- `User` -> `demo_user`
- `Partner name` -> `Omer`
- Toggle ON: **Memory Update**, **PB Enabled**
- Allow camera permission (or check **Sim Face** for deterministic playback)
- The `Local time` chip should be ticking; that clock is fed into the pipeline

LLM check (optional but recommended):
- Visit `/health/llm` in another tab.
- 200 OK = Gemini rewriter is live; 503 = rule-based floor will be used. **Either is fine** for the demo because Bonus 4 is exactly this latency-optimised fallback.

---

## 1) Opening (45 seconds, no clicks)

> "We built a multimodal AAC chatbot for partner-assisted communication. An AAC user
> with motor or speech impairments speaks through us; a partner asks them a question,
> and within 5 seconds we produce three identity-faithful, polarity-correct, grounded
> response options. Today I will show the seven-node pipeline running on a live
> browser, then walk through all five P4 bonuses."

Show the full UI once: video tile, sensor panel (HR, gaze, air-sign), partner input
box, and the three-option response panel.

---

## 2) Required Node Walk-Through (3 minutes)

### Turn 1 — memory grounding + face cue + multimodal mapping

1. **Sim Face** -> `smile`.
2. Type in partner box: `Are we still on for the movie tonight at 7?`
3. Click **Send**.

What to point out as the response appears:
- Three options appear inside ~2 seconds.
- The lead option is **agree-leaning + warm** (smile face -> positive polarity, warm tone).
- One of the options cites **"movie at 7"**; that is `today_plans` from STM, retrieved
  by the bucket selector and rerank stage.
- In the right-side log: you can see the bucket chosen (`today_plans` or `plans`),
  the evidence snippet, and the candidate generator output.

> "That is nodes 1 through 7 in one shot: face cue, multimodal mapping, parallel prep
> with router and memory, source-priority planner, bucket selector, retrieve-and-rerank
> with the groundedness guard, and the candidate generator with the rule-based
> floor and LLM rewriter."

### Turn 2 — polarity differential (gesture-driven)

1. **Sim Face** -> `shake_no`.
2. Same query: `Do you want a prescription reminder tonight?`
3. Click **Send**.

Point out: the lead is now **decline-leaning** ("Maybe not tonight", "Not right now,
thanks"). Same query, different gesture, polarity flips.

### Turn 3 — tone differential (face-only)

1. **Sim Face** -> `confused`.
2. Same query: `Are we still on for the movie tonight?`
3. Click **Send**.

Point out: lead is **clarify-first** ("Could you remind me what time?", "I want to
double-check the time"). Tone changed but the option is still grounded in memory.

---

## 3) Bonus Walk-Through (4–5 minutes — the headline section)

> "Five bonuses, demoed in order."

### Bonus 1 — Gaze-Based Retrieval Activation

**Setup**: in the sensor panel, set the **gaze target bucket** dropdown to `medical`.

1. Type: `What should I do for tonight?`
2. Click **Send**.

What to say while pointing at the response:
- "Without gaze, this question would route to `today_plans` or `routine`. Look at the
  log: the gazed bucket `medical` got a +0.30 score boost, so the lead option is the
  evening medication reminder. Gaze redirects retrieval."

### Bonus 2 — Vocal vs Air-Sign Conflict Resolution

**Setup**: turn the **Air Sign** input ON in the sensor panel and set it to a
deliberate negative gesture (e.g. `palm_down` or `index_left`). Keep face neutral.

1. Type a partner query that arrives with positive vocal tone in the demo file:
   `Are you ready for the movie?`
2. Click **Send**.

Point out:
- "Vocal channel says positive, spatial air-sign channel says negative. Our policy
  is **deliberate spatial wins**: the lead option is decline-leaning, and the log shows
  `conflict_resolved: spatial`."

### Bonus 3 — Acceptance-Weighted Bucket Priors

This one needs two turns to show the prior building up.

1. Type: `Want to order food tonight?` and **Send**.
2. From the three options, **click the food/dinner option** (e.g. "Yes, lets order
   from the usual place"). This calls `confirm_response` and increments
   `bucket_acceptance['food']`.
3. Type a slightly ambiguous follow-up: `What about later?`
4. Click **Send**.

Point out:
- "The second turn is genuinely ambiguous. The prior we built from the previous
  selection nudges the bucket selector toward `food` again; see the bucket score
  in the log; `food` gets a small additive prior. Selection feedback shapes future
  retrieval."

### Bonus 4 — Latency-Optimised Fallback (5-second SLA)

This bonus is **always** running; we just call attention to it.

1. Type: `Quick — yes or no, dinner at home tonight?` and **Send**.
2. In the response panel, point at the **Latency** badge.

Say:
- "Every turn races against a 5-second deadline. We try `gemini-2.5-flash-lite`
  first; if that has not returned by ~1.8 s we escalate to `gemini-2.5-flash`; if
  even that is slow, the rule-based floor (which is always ready immediately) is
  emitted. Across our 20-case eval the average was 1.8 s and p95 was 2.4 s; every
  case under 5 s, fallback engaged on 11 of 20."
- "If `/health/llm` returned 503 right now, the rule-based floor handled this
  whole demo; that is the point of the fallback."

### Bonus 5 — Online Index Update on Selection

1. Type: `Whats our plan for tonight?` and **Send**.
2. Click any one of the three options (do not pick the lead; pick the second).
3. Type the **same question again**: `Whats our plan for tonight?` and **Send**.

Point out:
- "On the second turn, the option you picked last time is boosted by +0.15 in the
  phrase bank score, so it surfaces earlier in the candidate pool. Over a session,
  the bank quietly tunes to the user's actual choices. In our 5-turn rollout we
  added 2 new phrases and boosted 3 existing ones."

---

## 4) Memory Write + Recall (1 minute)

Show the memory pipeline closing the loop.

1. Keep **Memory Update** ON.
2. Type: `Tomorrow we have project rehearsal at 5:15 PM.` and **Send**, then **confirm** any option.
3. Watch the **Memory Ack** chip: it should say something like
   `Memory Ack: added to next_days_plans`.
4. Now type: `What did I add for tomorrow at 5:15?` and **Send**.

Point out: the lead option recites the plan you just added; the memory write was
routed to the right bucket and is retrievable immediately.

---

## 5) Closing (30 seconds)

> "To summarize: 7-node training-free pipeline, 12-bucket memory schema, deadline-raced
> LLM rewriter on top of a rule-based floor, all 5 P4 bonuses live, average latency
> 1.8 s on 20 held-out cases, polarity adherence 95%, multimodal alignment 87.5%, and
> all face frames stay in the browser. The full report, slides, and source are in
> the submission zip and on GitHub. Questions?"

---

## Quick Reference — Mapping of Demo Steps to Required Components

| P4 requirement | Where it shows up in this demo |
|---|---|
| Multimodal input mapping | Turn 1 (face), Turn 2 (gesture), Bonus 1 (gaze), Bonus 2 (air-sign) |
| Partner-aware retrieval | Turn 1 (movie plan from STM), Turn 4 (Memory recall) |
| Identity-faithful generation | Three options every turn, polarity flip in Turn 2 |
| Memory update | Turn 4 (Memory Ack chip) |
| Evaluation/grounding | Latency badge, GroundednessGuard logged in side panel |
| Bonus 1: Gaze activation | Bonus 1 turn |
| Bonus 2: Conflict resolution | Bonus 2 turn |
| Bonus 3: Acceptance priors | Bonus 3 two-turn sequence |
| Bonus 4: Latency race | Always-on; called out on Bonus 4 turn |
| Bonus 5: Online index update | Bonus 5 two-turn sequence |

---

## If Something Breaks Mid-Demo

| Symptom | Recovery |
|---|---|
| No options appear | Click **START** again; ensure `User=demo_user`. |
| Camera black | Toggle **Sim Face** ON and pick a preset. |
| `/health/llm` is 503 | Do not fix it; explicitly call out Bonus 4 (rule-based floor). |
| Memory Ack missing | **Memory Update** checkbox must be ON before clicking the option. |
| Latency spike | Ask the audience to count to 5; it will still come back inside the SLA. |

---

## Recording the Demo Video (~5 minutes)

If using QuickTime or OBS, record this slimmed sequence:
1. (0:00–0:30) Opening: UI tour.
2. (0:30–1:30) Turns 1–3: required nodes, polarity and tone differentials.
3. (1:30–4:00) Bonuses 1 -> 5 in order, one turn each.
4. (4:00–4:45) Memory write and recall.
5. (4:45–5:00) Closing summary slide.
