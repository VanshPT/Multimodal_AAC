from django.test import SimpleTestCase

from home.aac.memory.store import memory_store
from home.aac.pipelines.nodes import candidate_generator_node, face_cue_node
from home.aac.service import confirm_response, start_session


class MemoryUpdateTests(SimpleTestCase):
    def test_memory_update_on_adds_recent_turn(self):
        session = start_session("demo_user")
        session_id = session["session_id"]
        response = confirm_response(
            session_id=session_id,
            partner_text="Are you ready for the movie?",
            selected_text="Yes, movie at seven still works for me.",
            final_text="Yes, movie at seven still works for me.",
            memory_update_on=True,
            face_signals={"smile_prob": 0.8},
        )
        self.assertEqual(response["memory_update_actions"]["stm"], "updated")
        self.assertIn(response["memory_update_actions"]["pb"]["action"], {"boost_existing_phrase", "add_new_phrase"})

    def test_partner_memory_statement_gets_bucketed(self):
        session = start_session("demo_user")
        session_id = session["session_id"]
        response = confirm_response(
            session_id=session_id,
            partner_text="Tomorrow we have project rehearsal at 5:15 PM",
            selected_text="Got it.",
            final_text="Got it, I noted that.",
            memory_update_on=True,
            face_signals={"smile_prob": 0.1, "confused_prob": 0.6},
        )
        partner_memory = response["memory_update_actions"]["partner_memory"]
        self.assertIn(partner_memory["action"], {"added", "already_present"})
        self.assertEqual(partner_memory["bucket"], "next_days_plans")
        state = memory_store.get_session(session_id)
        self.assertTrue(any("project rehearsal at 5:15 PM" in item for item in state.stm.get("next_days_plans", [])))
        self.assertIn("next_days_plans", response["memory_update_ack"])

    def test_face_detection_infers_when_confused_signal_present(self):
        signals = face_cue_node(camera_on=True, provided_face_signals={"smile_prob": 0.0, "confused_prob": 0.82, "neutral_prob": 0.18})
        self.assertIsNotNone(signals)
        self.assertTrue(signals["face_detected"])

    def test_binary_generation_balances_agree_and_decline(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        options = candidate_generator_node(
            state=state,
            partner_text="Do you want a prescription reminder tonight?",
            label="Contextual",
            evidence=[],
            guardrails={"mode": "grounded", "instruction": "none"},
            face_signals={"smile_prob": 0.2, "confused_prob": 0.2, "shake_score": 0.1, "nod_score": 0.1, "negative_prob": 0.1, "face_detected": True},
            pb_exemplars=[],
            partner_style_hint="casual",
        )
        joined = " ".join(options).lower()
        self.assertIn("yes", joined)
        self.assertTrue("not" in joined or "no" in joined or "pass" in joined)
