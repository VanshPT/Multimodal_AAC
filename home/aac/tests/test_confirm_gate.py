from django.test import SimpleTestCase

from home.aac.service import confirm_response, get_session_state, handle_partner_message, start_session


class ConfirmGateTests(SimpleTestCase):
    def test_final_output_requires_confirm_call(self):
        session = start_session("demo_user")
        session_id = session["session_id"]
        handle_partner_message(
            session_id=session_id,
            partner_text="Are we still on for tonight?",
            camera_on=False,
            face_signals=None,
            pb_enabled=True,
        )
        snapshot_before = get_session_state(session_id)
        self.assertIsNone(snapshot_before["final_output"])

        confirm_response(
            session_id=session_id,
            partner_text="Are we still on for tonight?",
            selected_text="Yes, still on.",
            final_text="Yes, still on.",
            memory_update_on=False,
            face_signals=None,
        )
        snapshot_after = get_session_state(session_id)
        self.assertEqual(snapshot_after["final_output"], "Yes, still on.")
