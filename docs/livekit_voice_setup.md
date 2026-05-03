# LiveKit Voice Setup

This AAC app now supports an optional LiveKit voice layer around the existing partner-input and confirm flow.

## What it does

- Press `SPEAK` to open a LiveKit microphone capture turn.
- Press `x` to stop listening.
- The final transcript is sent into the existing `/aac/api/partner_message/` flow, so you still get the same 3 response options and edit box.
- Press `Confirm` and the final confirmed AAC response is spoken aloud through a one-shot LiveKit TTS turn.

## Environment variables

Copy [.env.example](/C:/Users/Admin/Desktop/MultiModal_AAC/.env.example) into your real environment and set:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `AAC_INGEST_SECRET`
- `AAC_BACKEND_BASE_URL`

Optional model overrides:

- `AAC_LIVEKIT_AGENT_NAME`
- `AAC_LIVEKIT_STT_MODEL`
- `AAC_LIVEKIT_TTS_MODEL`
- `AAC_LIVEKIT_TTS_LANGUAGE`
- `AAC_LIVEKIT_LLM_MODEL`

## Install dependencies

```powershell
pip install -r requirements.txt
```

## Run Django

```powershell
python manage.py runserver
```

## Run the LiveKit worker

```powershell
python scripts/livekit_aac_agent.py dev
```

If the first run asks for model assets for the Silero VAD plugin, let it download them.

## Notes

- LiveKit Cloud credentials are required.
- With LiveKit Inference, STT and TTS models are billed through LiveKit Cloud, so separate Deepgram or Cartesia API keys are not required for this setup.
- The current AAC brain, camera logic, retrieval, and confirm flow are unchanged. LiveKit only wraps speech input and spoken output.
