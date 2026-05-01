import json
import os
import shutil
import sys
from pathlib import Path

import django

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "softgo.settings")
django.setup()

from home.aac.memory.store import MemoryStoreError, memory_store  # noqa: E402
from home.aac.service import (  # noqa: E402
    confirm_response,
    get_session_state,
    handle_partner_message,
    handle_speak_mode,
    set_active_partner,
    start_session,
)


def print_h(title):
    print(f"\n=== {title} ===")


def count_ltm_items(data):
    count = 0
    for _, value in data.items():
        if isinstance(value, dict):
            count += len(value)
        elif isinstance(value, list):
            count += len(value)
        else:
            count += 1
    return count


def main():
    base = ROOT / "data" / "synthetic_users" / "demo_user"
    ltm_path = base / "long_term_profile.json"
    stm_path = base / "short_term_memory.json"
    pb_path = base / "phrases.json"

    print_h("A1 start session + new session ids + loaded summary")
    s1 = start_session("demo_user")
    s2 = start_session("demo_user")
    print("session_id_1", s1["session_id"])
    print("session_id_2", s2["session_id"])
    print("session_ids_different", s1["session_id"] != s2["session_id"])
    print("loaded_summary", s2["loaded_summary"])
    print("active_session", get_session_state(s2["session_id"])["active_session"])

    print_h("A2 confirm gate (no final output before confirm)")
    handle_partner_message(s2["session_id"], "Are we still on for tonight?", False, None, True)
    before = get_session_state(s2["session_id"])
    print("final_output_before_confirm", before["final_output"])
    confirm_response(s2["session_id"], "Are we still on for tonight?", "Yes still on", "Yes still on", False, None)
    after = get_session_state(s2["session_id"])
    print("final_output_after_confirm", after["final_output"])

    print_h("A3/A6 explainability panel fields from backend payload")
    debug = handle_partner_message(s2["session_id"], "Can you remind me for tomorrow?", False, None, True)["debug_info"]
    print("debug_keys", sorted(debug.keys()))
    print("debug_payload", debug)

    print_h("A4/D16 speak groups + context-specific movie starters")
    speak = handle_speak_mode(s2["session_id"], False, None)
    print("speak_group_names", list(speak["grouped_suggestions"].keys()))
    today = speak["grouped_suggestions"]["Today"]
    movie_related = [line for line in today if "movie" in line.lower() or "6:30" in line]
    print("today_movie_related_examples", movie_related[:3])
    print("total_suggestions", sum(len(v) for v in speak["grouped_suggestions"].values()))

    print_h("B7 server-side no auto-finalize check")
    s3 = start_session("demo_user")
    handle_partner_message(s3["session_id"], "Do we need groceries?", False, None, True)
    print("snapshot_without_confirm", get_session_state(s3["session_id"]))

    print_h("B8 factual grounding rule + B9 do_not_say enforcement")
    personal = handle_partner_message(s3["session_id"], "What do you want for dinner after therapy?", False, None, True)
    print("options_after_safety_rules", personal["options"])
    unsupported = handle_partner_message(s3["session_id"], "Tell me about dad surgery history", False, None, True)
    print("unsupported_personal_query_options", unsupported["options"])

    print_h("C10 missing/corrupt JSON handling")
    backup = ltm_path.with_suffix(".json.bak")
    shutil.copy2(ltm_path, backup)
    ltm_path.write_text("{ bad json", encoding="utf-8")
    try:
        start_session("demo_user")
    except Exception as error:
        print("corrupt_json_error", str(error))
    finally:
        ltm_path.unlink(missing_ok=True)
        shutil.move(backup, ltm_path)

    print_h("C11 empty partner_text validation")
    try:
        handle_partner_message(s3["session_id"], "   ", False, None, True)
    except Exception as error:
        print("empty_partner_error", str(error))

    print_h("C12 zero useful chunks -> safe clarification")
    safe = handle_partner_message(s3["session_id"], "%%%% #### ????", False, None, True)
    print("safe_options", safe["options"])

    print_h("C13/C14 camera fallback + smoothing/coarse labels")
    no_face = handle_partner_message(s3["session_id"], "Are you free now?", True, None, True)
    smile = handle_partner_message(
        s3["session_id"], "Are you free now?", True, {"smile_prob": 0.95, "confused_prob": 0.02, "neutral_prob": 0.03}, True
    )
    confused = handle_partner_message(
        s3["session_id"], "Are you free now?", True, {"smile_prob": 0.05, "confused_prob": 0.9, "neutral_prob": 0.05}, True
    )
    print("camera_on_no_face_summary", no_face["debug_info"]["face_summary"])
    print("smile_option_1", smile["options"][0])
    print("confused_option_1", confused["options"][0])

    print_h("D15 synthetic data counts")
    ltm = json.loads(ltm_path.read_text(encoding="utf-8"))
    stm = json.loads(stm_path.read_text(encoding="utf-8"))
    pb = json.loads(pb_path.read_text(encoding="utf-8"))
    plans_total = len(stm.get("today_plans", [])) + len(stm.get("next_days_plans", [])) + len(stm.get("reminders", []))
    buckets = sorted(list({p.get("bucket_id") for p in pb.get("phrases", [])}))
    intents = sorted(list({p.get("intent") for p in pb.get("phrases", [])}))
    print("ltm_item_count", count_ltm_items(ltm))
    print("stm_plan_reminder_count", plans_total)
    print("pb_phrase_count", len(pb.get("phrases", [])))
    print("pb_bucket_count", len(buckets), buckets[:8])
    print("pb_intent_count", len(intents), intents)

    print_h("E17/E18 active partner stored and affects style")
    set_active_partner(s3["session_id"], "Rosa", "family")
    family_resp = handle_partner_message(s3["session_id"], "Are we still on for tonight?", False, None, True)
    set_active_partner(s3["session_id"], "Classmate", "general")
    general_resp = handle_partner_message(s3["session_id"], "Are we still on for tonight?", False, None, True)
    snap = get_session_state(s3["session_id"])
    print("active_partner_snapshot", snap["active_partner"])
    print("family_option_1", family_resp["options"][0])
    print("general_option_1", general_resp["options"][0])


if __name__ == "__main__":
    main()
