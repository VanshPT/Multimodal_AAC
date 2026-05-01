import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import django

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "softgo.settings")
django.setup()

from home.aac.evaluation.metrics import groundedness_score  # noqa: E402
from home.aac.memory.store import memory_store  # noqa: E402
from home.aac.pipelines.nodes import _ltm_chunks, _pb_chunks, _stm_chunks  # noqa: E402
from home.aac.service import confirm_response, handle_partner_message, handle_speak_mode, start_session  # noqa: E402


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def print_header(title: str):
    print(f"\n=== {title} ===")


def main():
    user_id = "demo_user"
    base = ROOT / "data" / "synthetic_users" / user_id
    ltm_path = base / "long_term_profile.json"
    stm_path = base / "short_term_memory.json"
    pb_path = base / "phrases.json"

    session = start_session(user_id)
    session_id = session["session_id"]
    state = memory_store.get_session(session_id)

    print_header("A1 flattened cards with bucket_id")
    print("LTM card:", _ltm_chunks(state)[0])
    print("STM card:", _stm_chunks(state)[0])
    print("PB card:", _pb_chunks(state)[0])

    query = "Can we review tomorrow schedule and groceries?"
    print_header("A2/A3/A4/A5 retrieval trace and fallback")
    resp = handle_partner_message(session_id, query, camera_on=False, face_signals=None, pb_enabled=True)
    print("search_order:", resp["debug_info"]["search_order"])
    print("sources_searched_in_order:", [step["pool"] for step in resp["retrieval_trace"]])
    if resp["retrieval_trace"]:
        first = resp["retrieval_trace"][0]
        print("before_refine_first_pool:", first["top_candidates_before_refine"][:2])
        print("refined_first_pool:", first["refined_evidence"][:2])
    if len(resp["retrieval_trace"]) > 1:
        second = resp["retrieval_trace"][1]
        print("refined_second_pool:", second["refined_evidence"][:2])
    print("merged_evidence_passed_to_generation:", resp["evidence_used"][:3])
    print("template_trace:", resp["template_trace"])
    print("options_count:", len(resp["options"]))

    print_header("B8 PB enabled vs disabled")
    enabled = handle_partner_message(session_id, "Are we still on for the movie tonight?", False, None, pb_enabled=True)
    disabled = handle_partner_message(session_id, "Are we still on for the movie tonight?", False, None, pb_enabled=False)
    print("enabled_option_1:", enabled["options"][0])
    print("disabled_option_1:", disabled["options"][0])

    print_header("C11/C12 face cues impact")
    off = handle_partner_message(session_id, "Are you free before 6:30?", camera_on=False, face_signals=None, pb_enabled=True)
    smile = handle_partner_message(
        session_id,
        "Are you free before 6:30?",
        camera_on=True,
        face_signals={"smile_prob": 0.9, "confused_prob": 0.05, "neutral_prob": 0.05},
        pb_enabled=True,
    )
    confused = handle_partner_message(
        session_id,
        "Are you free before 6:30?",
        camera_on=True,
        face_signals={"smile_prob": 0.1, "confused_prob": 0.85, "neutral_prob": 0.05},
        pb_enabled=True,
    )
    print("camera_off_option_1:", off["options"][0])
    print("camera_smile_option_1:", smile["options"][0])
    print("camera_confused_option_1:", confused["options"][0])

    print_header("D14 speak count/group proof")
    speak = handle_speak_mode(session_id, camera_on=False, face_signals=None)
    group_counts = {k: len(v) for k, v in speak["grouped_suggestions"].items()}
    print("groups:", list(speak["grouped_suggestions"].keys()))
    print("group_counts:", group_counts)
    print("total_suggestions:", sum(group_counts.values()))

    print_header("E16 OFF vs ON memory update behavior")
    stm_before_off = stm_path.read_text(encoding="utf-8")
    pb_before_off = pb_path.read_text(encoding="utf-8")
    confirm_response(
        session_id=session_id,
        partner_text="Do we need groceries?",
        selected_text="Yes, we should do groceries tonight.",
        final_text="Yes, we should do groceries tonight.",
        memory_update_on=False,
        face_signals=None,
    )
    stm_after_off = stm_path.read_text(encoding="utf-8")
    pb_after_off = pb_path.read_text(encoding="utf-8")
    print("memory_update_off_stm_changed:", stm_before_off != stm_after_off)
    print("memory_update_off_pb_changed:", pb_before_off != pb_after_off)

    stm_before_on = stm_path.read_text(encoding="utf-8")
    pb_before_on = pb_path.read_text(encoding="utf-8")
    on_result = confirm_response(
        session_id=session_id,
        partner_text="Do we need groceries?",
        selected_text="Yes, we should do groceries tonight.",
        final_text="Yes, we should do groceries tonight. Please add fruit too.",
        memory_update_on=True,
        face_signals={"smile_prob": 0.9, "confused_prob": 0.05, "neutral_prob": 0.05},
    )
    stm_after_on = stm_path.read_text(encoding="utf-8")
    pb_after_on = pb_path.read_text(encoding="utf-8")
    print("memory_update_on_stm_changed:", stm_before_on != stm_after_on)
    print("memory_update_on_pb_changed:", pb_before_on != pb_after_on)
    print("memory_update_actions:", on_result["memory_update_actions"])
    last_phrase = read_json(pb_path)["phrases"][-1]
    print("last_appended_phrase:", last_phrase)

    print_header("F20 groundedness example")
    one_option = resp["options"][0]
    evidence_texts = [e["text"] for e in resp["evidence_used"]]
    print("groundedness_score:", groundedness_score(one_option, evidence_texts))

    print_header("F22 latest log line keys")
    log_path = ROOT / "outputs" / "run_logs.jsonl"
    last_line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    parsed = json.loads(last_line)
    print("event:", parsed.get("event"))
    print("log_keys:", sorted(parsed.keys()))
    print("router_label:", parsed.get("router_label"))
    print("sources_searched_in_order:", parsed.get("sources_searched_in_order"))
    print("latency_ms:", parsed.get("latency_ms"))
    print("camera_used:", parsed.get("camera_used"))

    print_header("E18 ltm guard policy")
    print("ltm_update_policy:", read_json(ltm_path).get("ltm_update_policy"))


if __name__ == "__main__":
    main()
