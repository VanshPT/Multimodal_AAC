from django.test import Client, SimpleTestCase, override_settings

from home.aac.livekit_bridge import LiveKitVoiceBridge
from home.aac.service import start_session


class LiveKitVoiceBridgeTests(SimpleTestCase):
    def test_voice_turn_is_consumed_after_fetch(self):
        bridge = LiveKitVoiceBridge()
        record = bridge.create_turn(session_id="session-1", room_name="room-1")
        bridge.save_transcript(session_id="session-1", voice_turn_id=record.voice_turn_id, transcript="hello there")

        first = bridge.fetch_transcript(session_id="session-1", voice_turn_id=record.voice_turn_id, consume=True)
        second = bridge.fetch_transcript(session_id="session-1", voice_turn_id=record.voice_turn_id, consume=True)

        self.assertTrue(first["ready"])
        self.assertEqual(first["transcript"], "hello there")
        self.assertTrue(second["consumed"])


class LiveKitVoiceViewTests(SimpleTestCase):
    @override_settings(
        LIVEKIT_URL="",
        LIVEKIT_API_KEY="",
        LIVEKIT_API_SECRET="",
        AAC_INGEST_SECRET="",
    )
    def test_listen_token_endpoint_returns_config_error_when_livekit_missing(self):
        session = start_session("demo_user")
        client = Client()

        response = client.post(
            "/aac/api/livekit/listen_token/",
            data='{"session_id": "%s"}' % session["session_id"],
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertIn("LiveKit voice is not configured yet", payload["error"])
        self.assertIn("LIVEKIT_URL", payload["error"])
