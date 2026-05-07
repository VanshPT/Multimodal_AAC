# Multimodal AAC Chatbot — Recording Script

**Authors**: Syed Omer Shah, Vansh Thakkar
**Course**: CSE 635 — NLP and Text Mining (Spring 2026)
**Target runtime**: ~5 minutes 30 seconds.

Side-by-side script: each row is one beat. Left column = the click/keystroke, right column = the spoken line. Pause ~1 s after each click so the recording reads cleanly.

---

## Mac Screen Recording with Audio

### Option 1 — Built-in (mic only) — easiest

`Cmd + Shift + 5` opens the screen-recording toolbar.

1. Click **Record Selected Portion** (or Entire Screen).
2. Click **Options** -> **Microphone** = MacBook Pro Microphone (or your headset). Save to *Desktop*. Turn off *Show Floating Thumbnail*.
3. Drag the box around your Chrome window only (avoid VS Code overlap to keep file size small).
4. Click **Record**. Speak + drive the demo.
5. Stop with the menubar stop icon (or `Cmd + Ctrl + Esc`).

This records voice over screen but **not** system audio. For this demo that is fine.

### Option 2 — System audio + mic (only if you need UI sounds)

```bash
brew install blackhole-2ch
```

In **Audio MIDI Setup**, create a Multi-Output Device = `Built-in Output + BlackHole 2ch`. Pick that as system output. In QuickTime -> New Screen Recording -> Microphone = `BlackHole 2ch` to capture system; or use OBS to mix mic + BlackHole on separate tracks.

**Recommendation: just use Option 1.**

---

## Pre-Demo Sanity (60 seconds)

```
1. Server already running on :8000   (python manage.py runserver 8000)
2. Open Chrome -> http://127.0.0.1:8000/aac/
3. Click START
4. User dropdown   -> demo_user
5. Partner name    -> Omer
6. Memory Update   -> ON
7. PB Enabled      -> ON
8. Allow webcam (or check Sim Face)
9. Open second tab: http://127.0.0.1:8000/health/llm
   (verify {"ok": true, "model": "gemini-2.5-flash-lite"})
10. Cmd+Shift+5 -> Options -> Mic = MacBook Pro Mic -> Record
```

Also turn on **Do Not Disturb** (`Cmd+Option+,`) so notifications do not land on the recording.

---

## Section 1 — Opening (~30 s)

| UI / Action | Spoken Line |
|---|---|
| Camera shows your face; UI fully visible | "We built a multimodal AAC chatbot for partner-assisted communication. The user has motor or speech impairments; we generate three identity-faithful, polarity-correct, grounded reply options within a five-second SLA on every partner turn." |
| Hover over the sensor panel (HR, gaze, air-sign) | "Inputs are face emotion, hand gesture, gaze, heart rate, and an air-sign channel. Today I will show the seven-node pipeline plus all five P4 bonuses." |

## Section 2 — Required nodes (~90 s)

| UI / Action | Spoken Line |
|---|---|
| Sim Face -> `smile` | "Smile face." |
| Type `Are we still on for the movie tonight at 7?` -> **Send** | "Partner asks about a plan that lives in short-term memory." |
| Three options appear; point at lead | "Lead is agree-leaning, warm tone. Notice it cites *movie at seven*; that is the `today_plans` bucket retrieved by the bucket selector and reranked under the groundedness guard." |
| Sim Face -> `shake_no` | "Now I switch to a head-shake gesture." |
| Type `Do you want a prescription reminder tonight?` -> **Send** | "Same prompt class as before, different multimodal input." |
| Read new lead | "Polarity flips; lead is decline-leading. Same retrieval logic, gesture polarity overrides." |
| Sim Face -> `confused` | "Confused expression." |
| Type `Are we still on for the movie tonight?` -> **Send** | "Same memory, different face cue." |
| Read lead | "Tone shifts to clarify-first while staying grounded in the same memory." |

## Section 3 — Bonus 1: Gaze (~25 s)

| UI / Action | Spoken Line |
|---|---|
| In top bar, set **Sim Gaze** -> `medical` (live chip updates to `simulated: medical`) | "I am now telling the system the user is gazing at the medical tile." |
| Type `What should I do for tonight?` -> **Send** | "Open-ended question; default routing would prefer plans or routine." |
| Read lead | "Bonus 1: the gazed bucket gets a plus-zero-point-three score boost, so the lead surfaces the medication item from the medical bucket. Gaze redirects retrieval." |
| Reset **Sim Gaze** -> `(none)` before next section | — |

## Section 4 — Bonus 2: Vocal vs Air-Sign Conflict (~25 s)

| UI / Action | Spoken Line |
|---|---|
| Top bar: **Sim Vocal** -> `positive`, **Sim Air-Sign** -> `N (no)` | "Vocal channel says positive. The deliberate air-sign channel says N for No. They conflict." |
| Type `Are you ready for the movie?` -> **Send** | "Send it." |
| Read lead — should be decline-leaning | "Bonus 2: deliberate spatial wins. Lead is *Not ready, be there shortly*. The log shows `conflict_resolved: spatial`." |
| Reset **Sim Vocal** and **Sim Air-Sign** to `(none)` | — |

## Section 5 — Bonus 3: Acceptance Priors (~35 s)

| UI / Action | Spoken Line |
|---|---|
| Type `Want to order food tonight?` -> **Send** | "Clear question routes to food bucket." |
| **Click** the food/dinner option | "I select a food option. The system increments `bucket_acceptance['food']`." |
| Type `What about later?` -> **Send** | "Now an ambiguous follow-up." |
| Read lead | "Bonus 3: acceptance prior nudges the bucket selector toward food again. Selection feedback shapes future retrieval." |

## Section 6 — Bonus 4: Latency Race (~20 s)

| UI / Action | Spoken Line |
|---|---|
| Type `Quick — yes or no, dinner at home tonight?` -> **Send**, point at **Latency badge** | "Bonus 4: every turn races a five-second deadline. Flash-lite tries first; if slow we escalate to flash; the rule-based floor is always ready as the floor. Across the twenty-case eval: average one-point-eight seconds, p95 two-point-four seconds, fallback engaged on eleven of twenty." |

## Section 7 — Bonus 5: Online Phrase-Bank Update (~30 s)

| UI / Action | Spoken Line |
|---|---|
| Type `What's our plan for tonight?` -> **Send** | "Three options, varied phrasing." |
| **Click the second option** (not the lead) | "I pick the second one. The system boosts that phrase by plus-zero-point-one-five in the phrase bank." |
| Type the **same query again** -> **Send** | "Same question, second pass." |
| Point at re-ordered options | "Bonus 5: the phrase I picked last time now surfaces earlier. The bank quietly tunes to the user's actual choices." |

## Section 8 — Memory Write + Recall (~40 s)

| UI / Action | Spoken Line |
|---|---|
| Confirm **Memory Update** is ON. Type `Tomorrow we have project rehearsal at 5:15 PM.` -> **Send** -> confirm any option | "I am telling the system about a new plan for tomorrow." |
| Point at **Memory Ack** chip | "Memory ack: added to `next_days_plans`." |
| Type `What did I add for tomorrow at 5:15?` -> **Send** | "Now query it back." |
| Read lead | "The lead recites the plan I just added; the write was routed to the right bucket and is retrievable on the next turn." |

## Section 9 — Closing (~20 s)

| UI / Action | Spoken Line |
|---|---|
| Pull up Chrome tab with `/health/llm` showing `{"ok": true}` then back to UI | "Wrap-up: seven-node training-free pipeline, twelve-bucket memory schema, deadline-raced LLM rewriter on a rule-based floor, all five P4 bonuses live, polarity adherence ninety-five percent, multimodal alignment eighty-seven percent, every face frame stays in the browser. Full report, slides, and source are in the submission zip and on GitHub. Thanks." |
| Stop screen recording (menubar stop icon) | — |

---

## If Something Breaks Mid-Recording

| Symptom | Recovery |
|---|---|
| No options appear | Click **START** again; verify `User=demo_user`. |
| Camera black | Toggle **Sim Face** ON and pick a preset. |
| `/health/llm` is 503 | Do not fix it; explicitly call out Bonus 4 (rule-based floor handled the demo). |
| Memory Ack missing | **Memory Update** checkbox must be ON before clicking the option. |
| Latency spike | Wait it out; it will still come back inside the SLA. |

---

## Practice Pass

Run through Sections 2 and 3 once before pressing record. The Bonus 2 and Bonus 5 effects are subtle; you want to know exactly what the lead option will say so you can describe it crisply.
