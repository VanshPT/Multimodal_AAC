import json
from pathlib import Path
from typing import List

from django.conf import settings

from home.aac.evaluation.metrics import edit_distance_ratio, groundedness_score, summarize_metrics
from home.aac.service import confirm_response, handle_partner_message, handle_speak_mode, start_session


def _normal_queries() -> List[str]:
    return [
        "Are we still on for the movie tonight?",
        "What do you want for dinner after therapy?",
        "Can you remind me what your schedule is tomorrow?",
        "Do you want me to call your mom now?",
        "How are you feeling before class?",
    ]


def run_evaluation(user_id: str = "demo_user"):
    started = start_session(user_id)
    session_id = started["session_id"]

    normal_latencies = []
    grounded_scores = []
    effort = []
    transcript_lines = ["# Sample AAC Transcript", ""]

    for index, query in enumerate(_normal_queries(), start=1):
        result = handle_partner_message(
            session_id=session_id,
            partner_text=query,
            camera_on=False,
            face_signals=None,
        )
        options = result["options"]
        debug_info = result["debug_info"]
        evidence = [item["text"] for item in result["evidence_used"]]
        normal_latencies.append(debug_info["latency_ms"])
        for option in options:
            grounded_scores.append(groundedness_score(option, evidence))

        selected = options[0]
        final_text = selected if index % 2 else f"{selected} Please keep it brief."
        confirmed = confirm_response(
            session_id=session_id,
            partner_text=query,
            selected_text=selected,
            final_text=final_text,
            memory_update_on=True,
            face_signals=None,
        )
        effort.append(
            {
                "query": query,
                "edit_distance": edit_distance_ratio(selected, final_text),
                "accepted_without_edit": final_text == selected,
            }
        )
        transcript_lines.extend(
            [
                f"## Turn {index}",
                f"- Partner: {query}",
                f"- Candidate chosen: {selected}",
                f"- Final output: {confirmed['final_output']}",
                "",
            ]
        )

    speak_latencies = []
    smile_result = handle_speak_mode(session_id=session_id, camera_on=True, face_signals={"smile_prob": 0.8, "confused_prob": 0.1})
    confused_result = handle_speak_mode(session_id=session_id, camera_on=True, face_signals={"smile_prob": 0.1, "confused_prob": 0.8})
    speak_latencies.append(smile_result["debug_info"]["latency_ms"])
    speak_latencies.append(confused_result["debug_info"]["latency_ms"])

    smile_first = next(iter(smile_result["grouped_suggestions"].values()))[0]
    confused_first = next(iter(confused_result["grouped_suggestions"].values()))[0]
    multimodal_alignment = {
        "smile_example": smile_first,
        "confused_example": confused_first,
        "tone_changed": smile_first != confused_first,
    }

    summary = summarize_metrics(
        latencies_normal=normal_latencies,
        latencies_speak=speak_latencies,
        grounded_scores=grounded_scores,
        effort=effort,
    )
    summary["multimodal_alignment"] = multimodal_alignment

    outputs_dir = Path(settings.BASE_DIR) / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outputs_dir / "sample_transcripts.md").write_text("\n".join(transcript_lines), encoding="utf-8")
    return summary
