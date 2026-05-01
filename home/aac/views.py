import json
from pathlib import Path

from django.conf import settings
from django.http import FileResponse
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from home.aac.memory.store import MemoryStoreError, memory_store
from home.aac.llm import gemini_client
from home.aac.service import (
    confirm_response,
    get_session_state,
    handle_partner_message,
    handle_speak_mode,
    start_session,
)


@require_GET
def aac_session_page(request):
    return render(request, "home/aac_session.html", {"users": memory_store.list_users()})


@require_GET
def list_users_api(request):
    return JsonResponse({"users": memory_store.list_users()})


@require_POST
@csrf_exempt
def start_session_api(request):
    payload = json.loads(request.body or "{}")
    user_id = payload.get("user_id")
    if not user_id:
        return JsonResponse({"error": "user_id is required"}, status=400)
    try:
        session = start_session(user_id=user_id)
    except MemoryStoreError as error:
        return JsonResponse({"error": str(error)}, status=400)
    return JsonResponse(session)


@require_POST
@csrf_exempt
def partner_message_api(request):
    payload = json.loads(request.body or "{}")
    try:
        response = handle_partner_message(
            session_id=payload.get("session_id", ""),
            partner_text=payload.get("partner_text", ""),
            camera_on=bool(payload.get("camera_on", False)),
            face_signals=payload.get("face_signals"),
            pb_enabled=bool(payload.get("pb_enabled", True)),
            partner_name=payload.get("partner_name", ""),
            client_now=payload.get("client_now", ""),
        )
    except (ValueError, MemoryStoreError) as error:
        return JsonResponse({"error": str(error)}, status=400)
    return JsonResponse(response)


@require_POST
@csrf_exempt
def speak_mode_api(request):
    payload = json.loads(request.body or "{}")
    try:
        response = handle_speak_mode(
            session_id=payload.get("session_id", ""),
            camera_on=bool(payload.get("camera_on", False)),
            face_signals=payload.get("face_signals"),
            client_now=payload.get("client_now", ""),
        )
    except (ValueError, MemoryStoreError) as error:
        return JsonResponse({"error": str(error)}, status=400)
    return JsonResponse(response)


@require_POST
@csrf_exempt
def confirm_response_api(request):
    payload = json.loads(request.body or "{}")
    try:
        response = confirm_response(
            session_id=payload.get("session_id", ""),
            partner_text=payload.get("partner_text", ""),
            selected_text=payload.get("selected_text", ""),
            final_text=payload.get("final_text", ""),
            memory_update_on=bool(payload.get("memory_update_on", False)),
            face_signals=payload.get("face_signals"),
            query_id=payload.get("query_id", ""),
        )
    except (ValueError, MemoryStoreError) as error:
        return JsonResponse({"error": str(error)}, status=400)
    return JsonResponse(response)


@require_GET
def session_state_api(request):
    try:
        response = get_session_state(request.GET.get("session_id", ""))
    except (ValueError, MemoryStoreError) as error:
        return JsonResponse({"error": str(error)}, status=400)
    return JsonResponse(response)


@require_GET
def metrics_page(request):
    logs = _load_metrics_logs()
    normalized = []
    for row in logs[-200:]:
        normalized.append(
            {
                "timestamp": row.get("timestamp", "-"),
                "event": row.get("event", "-"),
                "partner_detected": row.get("partner_detected", "-"),
                "latency_ms": row.get("latency_ms", "-"),
                "groundedness_score": row.get("groundedness_score", "-"),
                "unsupported_claims_count": row.get("unsupported_claims_count", "-"),
                "sources_text": ", ".join(row.get("sources_searched_in_order") or row.get("sources") or []),
                "camera_face_text": f"{row.get('camera_used', '-')} / {row.get('face_signals_used', '-')}",
                "edit_distance": row.get("edit_distance", "-"),
                "acceptance": row.get("acceptance", "-"),
            }
        )
    return render(request, "home/metrics.html", {"rows": normalized})


@require_GET
def download_metrics_logs(request):
    path = Path(settings.BASE_DIR) / "outputs" / "run_logs.jsonl"
    if not path.exists():
        return JsonResponse({"error": "No logs found yet. Generate at least one request first."}, status=404)
    return FileResponse(path.open("rb"), as_attachment=True, filename="run_logs.jsonl")


@require_GET
def llm_health_api(request):
    result = gemini_client.health_ping()
    status = 200 if result.get("ok") else 503
    return JsonResponse(
        {
            "ok": bool(result.get("ok")),
            "model": result.get("model", ""),
            "latency_ms": result.get("latency_ms", 0),
            "error": result.get("error", ""),
        },
        status=status,
    )


def _load_metrics_logs():
    path = Path(settings.BASE_DIR) / "outputs" / "run_logs.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
