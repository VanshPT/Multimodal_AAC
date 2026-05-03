import json
import logging
import os
from typing import Optional

import aiohttp
from dotenv import load_dotenv
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, cli, inference, llm, utils
from livekit.plugins import silero


logger = logging.getLogger("aac-livekit-agent")
logging.basicConfig(level=logging.INFO)

load_dotenv()
load_dotenv(",env")

BACKEND_BASE_URL = os.getenv("AAC_BACKEND_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
INGEST_SECRET = os.getenv("AAC_INGEST_SECRET", os.getenv("INGEST_SECRET", "")).strip()
AGENT_NAME = os.getenv("AAC_LIVEKIT_AGENT_NAME", "aac-voice-router").strip() or "aac-voice-router"
STT_MODEL = os.getenv("AAC_LIVEKIT_STT_MODEL", "deepgram/nova-3").strip() or "deepgram/nova-3"
TTS_MODEL = os.getenv("AAC_LIVEKIT_TTS_MODEL", "cartesia/sonic-3").strip() or "cartesia/sonic-3"
TTS_LANGUAGE = os.getenv("AAC_LIVEKIT_TTS_LANGUAGE", "en").strip() or "en"
LLM_MODEL = os.getenv("AAC_LIVEKIT_LLM_MODEL", "google/gemini-2.5-flash").strip() or "google/gemini-2.5-flash"


async def send_transcript(session_id: str, voice_turn_id: str, transcript: str) -> None:
    url = f"{BACKEND_BASE_URL}/aac/api/livekit/ingest_transcript/"
    payload = {
        "session_id": session_id,
        "voice_turn_id": voice_turn_id,
        "transcript": transcript,
    }
    headers = {
        "Content-Type": "application/json",
        "X-INGEST-SECRET": INGEST_SECRET,
    }
    session = utils.http_context.http_session()
    timeout = aiohttp.ClientTimeout(total=20)
    async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
        body = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"Transcript ingest failed with HTTP {response.status}: {body[:200]}")


class AacVoiceAgent(Agent):
    def __init__(self, metadata: str) -> None:
        super().__init__(instructions="You are a thin AAC voice router. Never improvise or answer on your own.")
        self.meta_raw = metadata or "{}"
        self.session_id = ""
        self.voice_turn_id = ""
        self.mode = "capture"
        self.tts_text = ""
        self._pending_user_text: Optional[str] = None
        self._parse_metadata()

    def _parse_metadata(self) -> None:
        meta = json.loads(self.meta_raw or "{}")
        self.session_id = (meta.get("session_id") or "").strip()
        self.voice_turn_id = (meta.get("voice_turn_id") or "").strip()
        self.mode = (meta.get("mode") or "capture").strip().lower()
        self.tts_text = (meta.get("text") or "").strip()
        if not self.session_id:
            raise RuntimeError("Missing session_id in LiveKit job metadata.")

    async def on_enter(self):
        if self.mode == "tts" and self.tts_text:
            await self.session.say(self.tts_text, allow_interruptions=False)

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        if self.mode != "capture":
            return
        text = ""
        try:
            text = getattr(new_message, "text_content", "") or ""
            if callable(text):
                text = text()
        except Exception:
            text = getattr(new_message, "text", "") or ""
        self._pending_user_text = str(text).strip()

    async def llm_node(self, chat_ctx: llm.ChatContext, tools: list, model_settings):
        if self.mode != "capture":
            return
        transcript = (self._pending_user_text or "").strip()
        self._pending_user_text = None
        if not transcript:
            return
        await send_transcript(
            session_id=self.session_id,
            voice_turn_id=self.voice_turn_id,
            transcript=transcript,
        )
        logger.info("Stored transcript for session=%s voice_turn=%s", self.session_id, self.voice_turn_id)


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(model=STT_MODEL, language="en"),
        llm=inference.LLM(model=LLM_MODEL),
        tts=inference.TTS(model=TTS_MODEL, language=TTS_LANGUAGE),
        vad=ctx.proc.userdata["vad"],
        min_endpointing_delay=0.35,
        max_endpointing_delay=1.0,
        allow_interruptions=False,
        preemptive_generation=False,
    )

    await session.start(
        agent=AacVoiceAgent(metadata=ctx.job.metadata),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
