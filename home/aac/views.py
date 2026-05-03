import json
import os
from pathlib import Path
from uuid import uuid4

from django.conf import settings
from django.http import FileResponse
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from home.aac.memory.store import MemoryStoreError, memory_store
from home.aac.llm import gemini_client
from home.aac.livekit_bridge import livekit_voice_bridge
from home.aac.service import (
    _session_or_raise,
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


@require_POST
@csrf_exempt
def livekit_listen_token_api(request):
    payload = json.loads(request.body or "{}")
    session_id = (payload.get("session_id") or "").strip()
    if not session_id:
        return JsonResponse({"error": "session_id is required"}, status=400)
    try:
        _session_or_raise(session_id)
    except (ValueError, MemoryStoreError) as error:
        return JsonResponse({"error": str(error)}, status=400)
    config_error = _livekit_config_error(require_ingest_secret=True)
    if config_error:
        return JsonResponse({"error": config_error, "manual_setup_hint": _livekit_manual_setup_hint()}, status=503)
    livekit_items = _load_livekit_sdk()
    if "error" in livekit_items:
        return JsonResponse({"error": livekit_items["error"]}, status=503)
    room_name = f"aac-listen-{session_id[:8]}-{uuid4().hex[:8]}"
    voice_turn = livekit_voice_bridge.create_turn(session_id=session_id, room_name=room_name)
    metadata = json.dumps(
        {
            "session_id": session_id,
            "voice_turn_id": voice_turn.voice_turn_id,
            "mode": "capture",
        }
    )
    token = _build_livekit_token(
        access_token_cls=livekit_items["AccessToken"],
        video_grants_cls=livekit_items["VideoGrants"],
        room_configuration_cls=livekit_items["RoomConfiguration"],
        room_agent_dispatch_cls=livekit_items["RoomAgentDispatch"],
        room_name=room_name,
        identity=f"aac-web-listen-{voice_turn.voice_turn_id[:8]}",
        participant_name="AAC Voice Capture",
        metadata=metadata,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
        agent_name=settings.AAC_LIVEKIT_AGENT_NAME,
    )
    return JsonResponse(
        {
            "url": settings.LIVEKIT_URL,
            "room_name": room_name,
            "token": token,
            "voice_turn_id": voice_turn.voice_turn_id,
            "agent_name": settings.AAC_LIVEKIT_AGENT_NAME,
        }
    )


@require_GET
def livekit_transcript_api(request):
    session_id = (request.GET.get("session_id") or "").strip()
    voice_turn_id = (request.GET.get("voice_turn_id") or "").strip()
    if not session_id or not voice_turn_id:
        return JsonResponse({"error": "session_id and voice_turn_id are required"}, status=400)
    try:
        _session_or_raise(session_id)
    except (ValueError, MemoryStoreError) as error:
        return JsonResponse({"error": str(error)}, status=400)
    payload = livekit_voice_bridge.fetch_transcript(session_id=session_id, voice_turn_id=voice_turn_id, consume=True)
    return JsonResponse(payload)


@require_POST
@csrf_exempt
def livekit_ingest_transcript_api(request):
    if request.headers.get("X-INGEST-SECRET", "") != settings.AAC_INGEST_SECRET:
        return JsonResponse({"error": "Bad ingest secret"}, status=403)
    payload = json.loads(request.body or "{}")
    session_id = (payload.get("session_id") or "").strip()
    voice_turn_id = (payload.get("voice_turn_id") or "").strip()
    transcript = (payload.get("transcript") or "").strip()
    if not session_id or not voice_turn_id or not transcript:
        return JsonResponse({"error": "session_id, voice_turn_id, and transcript are required"}, status=400)
    try:
        _session_or_raise(session_id)
    except (ValueError, MemoryStoreError) as error:
        return JsonResponse({"error": str(error)}, status=400)
    record = livekit_voice_bridge.save_transcript(
        session_id=session_id,
        voice_turn_id=voice_turn_id,
        transcript=transcript,
    )
    if not record:
        return JsonResponse({"error": "Unknown or expired voice turn."}, status=404)
    return JsonResponse({"ok": True, "voice_turn_id": voice_turn_id, "transcript": record.transcript})


@require_POST
@csrf_exempt
def livekit_tts_token_api(request):
    payload = json.loads(request.body or "{}")
    session_id = (payload.get("session_id") or "").strip()
    speak_text = (payload.get("text") or "").strip()
    if not session_id:
        return JsonResponse({"error": "session_id is required"}, status=400)
    if not speak_text:
        return JsonResponse({"error": "text is required"}, status=400)
    try:
        _session_or_raise(session_id)
    except (ValueError, MemoryStoreError) as error:
        return JsonResponse({"error": str(error)}, status=400)
    config_error = _livekit_config_error(require_ingest_secret=False)
    if config_error:
        return JsonResponse({"error": config_error, "manual_setup_hint": _livekit_manual_setup_hint()}, status=503)
    livekit_items = _load_livekit_sdk()
    if "error" in livekit_items:
        return JsonResponse({"error": livekit_items["error"]}, status=503)
    room_name = f"aac-tts-{session_id[:8]}-{uuid4().hex[:8]}"
    metadata = json.dumps(
        {
            "session_id": session_id,
            "mode": "tts",
            "text": speak_text,
        }
    )
    token = _build_livekit_token(
        access_token_cls=livekit_items["AccessToken"],
        video_grants_cls=livekit_items["VideoGrants"],
        room_configuration_cls=livekit_items["RoomConfiguration"],
        room_agent_dispatch_cls=livekit_items["RoomAgentDispatch"],
        room_name=room_name,
        identity=f"aac-web-tts-{uuid4().hex[:8]}",
        participant_name="AAC Voice Output",
        metadata=metadata,
        can_publish=False,
        can_subscribe=True,
        can_publish_data=False,
        agent_name=settings.AAC_LIVEKIT_AGENT_NAME,
    )
    return JsonResponse(
        {
            "url": settings.LIVEKIT_URL,
            "room_name": room_name,
            "token": token,
            "agent_name": settings.AAC_LIVEKIT_AGENT_NAME,
        }
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


def _livekit_config_error(require_ingest_secret: bool) -> str:
    missing = []
    for key in ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]:
        if not getattr(settings, key, ""):
            missing.append(key)
    if require_ingest_secret and not getattr(settings, "AAC_INGEST_SECRET", ""):
        missing.append("AAC_INGEST_SECRET")
    if not missing:
        return ""
    return f"LiveKit voice is not configured yet. Missing: {', '.join(missing)}"


def _livekit_manual_setup_hint() -> str:
    return (
        "Set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and AAC_INGEST_SECRET in your environment, "
        "then run the AAC LiveKit worker script."
    )


def _load_livekit_sdk():
    try:
        from livekit.api import AccessToken, RoomAgentDispatch, RoomConfiguration, VideoGrants
    except Exception as error:
        return {"error": f"LiveKit Python SDK is not installed: {error}"}
    return {
        "AccessToken": AccessToken,
        "RoomAgentDispatch": RoomAgentDispatch,
        "RoomConfiguration": RoomConfiguration,
        "VideoGrants": VideoGrants,
    }


def _build_livekit_token(
    *,
    access_token_cls,
    video_grants_cls,
    room_configuration_cls,
    room_agent_dispatch_cls,
    room_name: str,
    identity: str,
    participant_name: str,
    metadata: str,
    can_publish: bool,
    can_subscribe: bool,
    can_publish_data: bool,
    agent_name: str,
) -> str:
    token = (
        access_token_cls(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(participant_name)
        .with_metadata(metadata)
        .with_grants(
            video_grants_cls(
                room_join=True,
                room=room_name,
                can_publish=can_publish,
                can_subscribe=can_subscribe,
                can_publish_data=can_publish_data,
            )
        )
        .with_room_config(
            room_configuration_cls(
                agents=[room_agent_dispatch_cls(agent_name=agent_name, metadata=metadata)]
            )
        )
    )
    return token.to_jwt()
