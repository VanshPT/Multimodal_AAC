# Demo Script

## 1) Setup
1. Run `python scripts/generate_synthetic_user.py`.
2. Run server: `python manage.py runserver`.
3. Open `http://127.0.0.1:8000/aac/`.

## 2) Start session
1. Choose `demo_user` from dropdown.
2. Press `START`.
3. Confirm session id appears.

## 3) Scenario A - Partner message (normal mode)
1. Keep camera OFF.
2. Enter partner message: "Are we still on for the movie tonight?"
3. Press `Send Partner Message`.
4. Verify exactly 3 response cards appear.
5. Click `Select` on one card.
6. Optionally edit text in custom box.
7. Press `Confirm (✓)`.
8. Verify `Final Output` and transcript update.

## 4) Scenario B - Speak mode
1. Press `SPEAK`.
2. Verify 20-30 suggestions grouped by:
   - Today
   - Next few days
   - Reminders/tasks
   - People/topics
3. Select one, optionally edit, and confirm.
4. Verify final output + transcript updated.

## 5) Scenario C - Minimal inputs
1. Keep camera OFF or block camera permission.
2. Send partner message.
3. Verify system still returns 3 options and debug shows no face cues.

## 6) Scenario D - Camera ON
1. Turn camera ON.
2. If face detected, send a partner message and inspect debug panel.
3. Verify debug shows camera used and face summary.
4. If no face detected, verify "Face not detected" and pipeline still works.

## 7) Memory update behavior
1. Turn `Memory Update` ON.
2. Generate and confirm responses.
3. Verify `short_term_memory.json` and `phrases.json` get updated after confirmations.

## 8) Metrics run
1. Run `python scripts/run_evaluation.py`.
2. Verify outputs:
   - `outputs/metrics_summary.json`
   - `outputs/run_logs.jsonl`
   - `outputs/sample_transcripts.md`
