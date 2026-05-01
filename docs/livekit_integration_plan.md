# LiveKit Integration Plan (Future Phase)

This project currently supports typed partner input and confirmation-gated outputs. LiveKit speech-to-speech is intentionally deferred. This plan defines how to integrate LiveKit later without changing core retrieval logic.

## Integration goals
- Keep retrieval agent as source of suggestions.
- Preserve confirmation gate: no auto-speaking before user confirm.
- Add real-time audio streams for partner speech input and AAC output.

## Proposed flow
1. **LiveKit STT ingestion**
   - Partner audio stream -> LiveKit STT worker.
   - Partial + final transcript events emitted to Django websocket/API.
2. **Normal pipeline invocation**
   - On final STT segment, invoke existing normal mode pipeline:
     - face cues optional
     - retrieval fallback
     - exactly 3 options
3. **User confirmation UI**
   - Render options in AAC UI.
   - User selects/edits and confirms.
4. **LiveKit TTS output**
   - On confirm, send final text to LiveKit TTS service.
   - Play synthesized audio to room participants.
5. **Memory updates**
   - Reuse existing memory update subgraph when toggle is ON.

## Components to add
- LiveKit room/session manager service.
- Websocket bridge for low-latency transcripts and option updates.
- TTS queue to avoid overlapping utterances.
- Optional barge-in handler (cancel previous TTS when new confirm occurs).

## Safety and UX constraints
- Never auto-speak pipeline candidates.
- If STT confidence is low, prompt a clarification candidate.
- Keep textual final output visible even when TTS is active.

## Validation checklist for future phase
- STT transcript correctly enters normal pipeline.
- 3 options generated under real-time conditions.
- Confirmed text reliably triggers TTS in LiveKit.
- Latency from final STT segment to option rendering remains acceptable.
