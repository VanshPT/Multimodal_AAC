You are helping maintain and harden a multimodal AAC chatbot built on Django.

Project context:
- Repo: MultiModal_AAC
- Main UI: /aac/
- Core goal: generate 3 grounded AAC response options using memory + partner context + face/gesture cues.
- User persona in demo data: Akash (AAC user), partner commonly set to Omer.

What is already implemented:
1) Memory architecture
- LTM: long_term_profile.json
- STM: short_term_memory.json
- PB: phrases.json
- Session state tracks retrieval traces, options, confirmations, metrics.

2) Generation flow
- Router label: Personal / Contextual / Open-domain
- Source planning and retrieval over STM/LTM/PB
- Groundedness guard
- Candidate generation + post-processing
- Confirm gate to finalize output

3) Memory update behavior
- On confirm (if Memory Update ON):
  - PB phrase boost/add
  - STM recent turns + situation hints
  - Partner message extraction into bucketed STM memory
  - UI shows memory acknowledgement

4) Face + gesture support
- Supports smile/confused/neutral and nod/shake/negative scores.
- Binary prompts (yes/no style) now return mixed options (agree + decline + clarify), not only agreement.
- Face cues influence option ordering:
  - nod -> agree-first
  - shake/negative -> decline-first
  - confused -> clarify-first

5) UI instrumentation
- Debug panel includes face summary, nod/shake/negative scores, retrieval info.
- Sim Face presets include smile, confused, neutral, nod_yes, shake_no.
- Live local clock shown in UI and sent to backend (`client_now`) so runtime time context is available.

Current priorities for you:
1) Stabilize binary response quality
- Ensure 3 options are always semantically distinct.
- Preserve grounding while offering polarity variety.
- Avoid robotic repetitive patterns.

2) Improve real webcam nod/shake detection
- Calibrate thresholds for pitch/yaw range.
- Add temporal smoothing and cooldown so accidental micro-movements do not trigger false nod/shake.
- Keep simulation deterministic for testing.

3) Strengthen memory correctness
- Prevent weak or noisy partner text from polluting memory buckets.
- Add confidence filters and dedup improvements.
- Ensure retrieval of newly inserted memory in follow-up turns.

4) Improve explainability for testers
- Optionally add provenance badges (e.g., STM/LTM/PB source IDs).
- Keep debug output concise but actionable.

Testing expectations:
- `python manage.py test home.aac.tests` passes.
- Manual tests in /aac/ cover:
  - memory grounding
  - memory update and recall
  - same query under different face cues changes tone/order
  - binary prompts contain agree/decline/clarify set.

Important constraints:
- Do not remove confirm gate behavior.
- Do not regress memory update acknowledgements.
- Keep fallback behavior safe when detector APIs are unavailable.
- Keep code readable and deterministic where possible.

Recommended immediate checks:
1) Query: "Do you want a prescription reminder tonight?"
   - Test with nod_yes vs shake_no vs confused.
2) Insert memory:
   - "Tomorrow we have project rehearsal at 5:15 PM."
   - confirm
   - ask follow-up retrieval query.
3) Verify debug panel and metrics still populate.

If you detect regressions:
- Prioritize correctness + grounding over creativity.
- Add/adjust tests before broad refactors.
