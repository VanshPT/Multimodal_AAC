from django.test import SimpleTestCase

from home.aac.memory.store import memory_store
from home.aac.llm import gemini_client
from home.aac.pipelines.nodes import _is_binary_prompt, candidate_generator_node, face_cue_node
from home.aac.pipelines.normal_pipeline import run_normal_pipeline
from home.aac.pipelines.speak_pipeline import run_speak_pipeline
from home.aac.types import RetrievalEvidence
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

    def test_hand_gesture_affects_face_signal_interpretation(self):
        signals = face_cue_node(
            camera_on=True,
            provided_face_signals={
                "smile_prob": 0.1,
                "confused_prob": 0.1,
                "neutral_prob": 0.8,
                "hand_detected": True,
                "hand_gesture_label": "Thumb_Up",
                "hand_gesture_score": 0.9,
            },
        )
        self.assertIsNotNone(signals)
        self.assertEqual(signals["hand_gesture_label"], "Thumb_Up")
        self.assertGreaterEqual(signals["nod_score"], 0.55)

    def test_camera_off_generation_returns_tone_diverse_grounded_options(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        evidence = [
            RetrievalEvidence(pool="STM", bucket_id="plans", text="today plan: Movie with Omer at 7:00 PM at Regal North screen 6", score=0.92),
            RetrievalEvidence(pool="STM", bucket_id="plans", text="topic hint: meet before 6:30 at the campus bus stop", score=0.71),
        ]
        options = candidate_generator_node(
            state=state,
            partner_text="Are we still on for the movie tonight at 7?",
            label="Contextual",
            evidence=evidence,
            guardrails={"mode": "grounded", "instruction": "movie at 7 PM"},
            face_signals=None,
            pb_exemplars=[],
            partner_style_hint="casual",
        )
        self.assertEqual(len(options), 3)
        joined = " ".join(options).lower()
        self.assertIn("movie", joined)
        self.assertTrue(any("confirm" in option.lower() or "detail" in option.lower() for option in options))

    def test_negative_gesture_produces_diverse_decline_options(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        evidence = [
            RetrievalEvidence(pool="STM", bucket_id="plans", text="today plan: Therapy exercises at 4:30 PM", score=0.91),
            RetrievalEvidence(pool="STM", bucket_id="plans", text="today plan: Movie with Omer at 7:00 PM at Regal North screen 6", score=0.84),
        ]
        options = candidate_generator_node(
            state=state,
            partner_text="Do you remember our plans today?",
            label="Contextual",
            evidence=evidence,
            guardrails={"mode": "grounded", "instruction": "therapy at 4:30 PM"},
            face_signals={"face_detected": True, "hand_detected": True, "hand_gesture_label": "Thumb_Down", "hand_gesture_score": 0.9},
            pb_exemplars=[],
            partner_style_hint="casual",
        )
        self.assertEqual(len(options), 3)
        self.assertEqual(options[0], "No.")
        self.assertIn("do not remember", options[1].lower())
        self.assertIn("4:30", options[2])

    def test_head_shake_binary_prompt_includes_negative_response(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        evidence = [RetrievalEvidence(pool="STM", bucket_id="plans", text="today plan: Movie with Omer at 7:00 PM at Regal North screen 6", score=0.88)]
        options = candidate_generator_node(
            state=state,
            partner_text="Are we going to the movie today?",
            label="Contextual",
            evidence=evidence,
            guardrails={"mode": "grounded", "instruction": "movie at 7 PM"},
            face_signals={"face_detected": True, "shake_score": 0.85, "negative_prob": 0.25},
            pb_exemplars=[],
            partner_style_hint="casual",
        )
        self.assertEqual(options[0], "No.")
        self.assertTrue(any("do not remember" in option.lower() or "not fully sure" in option.lower() for option in options[1:]))

    def test_head_nod_binary_prompt_keeps_positive_option(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        evidence = [RetrievalEvidence(pool="STM", bucket_id="plans", text="today plan: Therapy exercises at 4:30 PM", score=0.9)]
        options = candidate_generator_node(
            state=state,
            partner_text="Do you remember the therapy plan?",
            label="Contextual",
            evidence=evidence,
            guardrails={"mode": "grounded", "instruction": "therapy at 4:30 PM"},
            face_signals={"face_detected": True, "nod_score": 0.84},
            pb_exemplars=[],
            partner_style_hint="casual",
        )
        self.assertIn("yes", options[0].lower())

    def test_unsure_cue_binary_prompt_prefers_clarification(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        evidence = [RetrievalEvidence(pool="STM", bucket_id="plans", text="today plan: Therapy exercises at 4:30 PM", score=0.9)]
        options = candidate_generator_node(
            state=state,
            partner_text="Do you remember our plans today?",
            label="Contextual",
            evidence=evidence,
            guardrails={"mode": "grounded", "instruction": "therapy at 4:30 PM"},
            face_signals={"face_detected": True, "confused_prob": 0.78},
            pb_exemplars=[],
            partner_style_hint="casual",
        )
        self.assertIn("not sure", options[0].lower())

    def test_recall_question_is_treated_as_binary_prompt(self):
        self.assertTrue(_is_binary_prompt("u know our plans for today?"))
        self.assertTrue(_is_binary_prompt("Do you remember our plans today?"))

    def test_hii_bro_is_treated_as_short_greeting(self):
        from home.aac.pipelines.nodes import is_short_greeting

        self.assertTrue(is_short_greeting("Hii bro"))

    def test_confirm_greeting_skips_memory_update(self):
        session = start_session("demo_user")
        session_id = session["session_id"]
        response = confirm_response(
            session_id=session_id,
            partner_text="Hii bro",
            selected_text="Hi, good to see you.",
            final_text="Hi, good to see you.",
            memory_update_on=True,
            face_signals={"face_detected": True},
        )
        self.assertEqual(response["memory_update_actions"]["stm"], "skipped_trivial_greeting")
        self.assertEqual(response["memory_update_actions"]["pb"]["action"], "skipped")
        self.assertEqual(response["memory_update_actions"]["partner_memory"]["action"], "skipped")
        self.assertIn("No memory update needed", response["memory_update_ack"])

    def test_greeting_options_ignore_decline_tone_override(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        options = candidate_generator_node(
            state=state,
            partner_text="Hii bro",
            label="Contextual",
            evidence=[],
            guardrails={"mode": "grounded", "instruction": "none"},
            face_signals={"face_detected": True, "shake_score": 0.88, "negative_prob": 0.1},
            pb_exemplars=[],
            partner_style_hint="casual",
        )
        joined = " ".join(options).lower()
        self.assertIn("hi", joined)
        self.assertNotIn("do not want", joined)
        self.assertNotIn("some other time", joined)

    def test_plan_change_prompt_returns_exact_negotiation_options(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        evidence = [
            RetrievalEvidence(pool="STM", bucket_id="plans", text="today plan: Movie with Omer at 7:00 PM at Regal North screen 6", score=0.94)
        ]
        options = candidate_generator_node(
            state=state,
            partner_text="u know we had plans for movie at 7? lets go tomorrow",
            label="Contextual",
            evidence=evidence,
            guardrails={"mode": "grounded", "instruction": "movie at 7 PM"},
            face_signals={"face_detected": True, "hand_gesture_label": "Thumb_Down", "hand_gesture_score": 0.8},
            pb_exemplars=[],
            partner_style_hint="casual",
        )
        self.assertEqual(
            options,
            ["No, let us go at 7 PM.", "No, let us go sometime else.", "Yes, it is fine with me."],
        )

    def test_later_today_query_keeps_grounded_memory_option(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        state.stm["today_plans"] = [
            "CSE 635 project check-in at 2:00 PM",
            "Therapy exercises at 4:30 PM",
            "Movie with Omer at 7:00 PM at Regal North screen 6",
            "Take evening medication at 8:30 PM",
            "Call Vansh before 9:30 PM about slides",
        ]
        evidence = [
            RetrievalEvidence(pool="STM", bucket_id="plans", text="today plan: CSE 635 project check-in at 2:00 PM", score=0.95),
            RetrievalEvidence(pool="STM", bucket_id="plans", text="today plan: Therapy exercises at 4:30 PM", score=0.93),
            RetrievalEvidence(pool="STM", bucket_id="plans", text="today plan: Take evening medication at 8:30 PM", score=0.85),
        ]
        options = candidate_generator_node(
            state=state,
            partner_text="What do I have later today?",
            label="Contextual",
            evidence=evidence,
            guardrails={"mode": "grounded", "instruction": "therapy at 4:30 PM | medication at 8:30 PM"},
            face_signals={"face_detected": True, "confused_prob": 0.52, "shake_score": 0.12, "negative_prob": 0.08},
            pb_exemplars=[],
            partner_style_hint="casual",
        )
        joined = " ".join(options).lower()
        self.assertIn("you have", options[0].lower())
        self.assertTrue("2:00" in joined or "check-in" in joined)
        self.assertTrue("4:30" in joined or "therapy" in joined)
        self.assertTrue("8:30" in joined or "medication" in joined)
        self.assertTrue("7:00" in joined or "movie" in joined)
        self.assertNotIn("let us keep the original time", joined)
        self.assertNotIn("partner asked", joined)

    def test_vansh_project_query_prefers_project_checkin_over_movie_noise(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        state.stm["active_partner"] = {"person_id": "vansh", "name": "Vansh", "relation": "friend"}
        state.stm["today_plans"] = [
            "CSE 635 project check-in at 2:00 PM",
            "Movie with Omer at 7:00 PM at Regal North screen 6",
        ]
        state.stm["next_days_plans"] = ["Tomorrow we have project rehearsal at 5:15 PM"]
        state.stm["recent_turns"] = [
            {"partner": "Are you ready for the movie?", "response": "Yes, movie at seven still works for me."},
            {"partner": "Tomorrow we have project rehearsal at 5:15 PM.", "response": "Yes, rehearsal works."},
        ]
        result = run_normal_pipeline(
            state=state,
            partner_text="Are we still okay for the project check-in tomorrow?",
            camera_on=True,
            provided_face_signals={"face_detected": True, "confused_prob": 0.05, "negative_prob": 0.05, "nod_score": 0.0, "shake_score": 0.0},
            pb_enabled=True,
        )
        joined = " ".join(result.options).lower()
        self.assertTrue("check-in" in joined or "2:00" in joined)
        self.assertNotIn("movie", joined)

    def test_confirm_response_moves_today_plan_to_tomorrow_when_final_text_reschedules(self):
        session = start_session("demo_user")
        session_id = session["session_id"]
        state = memory_store.get_session(session_id)
        state.stm["today_plans"] = [
            "CSE 635 project check-in at 2:00 PM",
            "Therapy exercises at 4:30 PM",
            "Movie with Omer at 7:00 PM at Regal North screen 6",
        ]
        response = confirm_response(
            session_id=session_id,
            partner_text="Do you remember our movie plan today?",
            selected_text="No, let us go some other time. 7:00 PM today.",
            final_text="No, let us go some other time. Maybe tomorrow.",
            memory_update_on=True,
            face_signals={"face_detected": True, "shake_score": 0.88},
        )
        self.assertEqual(response["memory_update_actions"]["partner_memory"]["action"], "rescheduled_to_future")
        state = memory_store.get_session(session_id)
        self.assertFalse(any("movie with omer at 7:00 pm" in item.lower() for item in state.stm.get("today_plans", [])))
        self.assertTrue(any("tomorrow: movie with omer at 7:00 pm" in item.lower() for item in state.stm.get("next_days_plans", [])))

    def test_confirm_response_affirming_tomorrow_updates_plan_to_future(self):
        session = start_session("demo_user")
        session_id = session["session_id"]
        state = memory_store.get_session(session_id)
        state.stm["today_plans"] = [
            "CSE 635 project check-in at 2:00 PM",
            "Therapy exercises at 4:30 PM",
            "Movie with Omer at 7:00 PM at Regal North screen 6",
        ]
        response = confirm_response(
            session_id=session_id,
            partner_text="You know we had plans for the movie at 7? Let's go tomorrow.",
            selected_text="Yes, it is fine with me.",
            final_text="Yes, it is fine with me.",
            memory_update_on=True,
            face_signals={"face_detected": True, "nod_score": 0.86},
        )
        self.assertIn(response["memory_update_actions"]["partner_memory"]["action"], {"rescheduled_to_future", "agreed_future_plan"})
        state = memory_store.get_session(session_id)
        self.assertTrue(any("tomorrow: movie with omer at 7:00 pm" in item.lower() for item in state.stm.get("next_days_plans", [])))

    def test_speak_pipeline_marks_llm_enabled_when_refiner_succeeds(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        original_enabled = gemini_client.enabled
        original_method = gemini_client.generate_speak_suggestions
        try:
            gemini_client.enabled = True

            def fake_generate(grouped, style, face_summary):
                return grouped, "gemini-test-model", ""

            gemini_client.generate_speak_suggestions = fake_generate
            result = run_speak_pipeline(
                state=state,
                camera_on=False,
                provided_face_signals=None,
            )
            self.assertTrue(result.debug_info.llm_enabled)
            self.assertEqual(result.debug_info.model_used, "gemini-test-model")
            self.assertIn("SpeakLLMRefinerNode", result.debug_info.node_trace)
        finally:
            gemini_client.enabled = original_enabled
            gemini_client.generate_speak_suggestions = original_method

    def test_medication_status_query_stays_uncertain_and_never_affirmative(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        state.stm["today_plans"] = [
            "Take evening medication at 8:30 PM",
            "Movie with Omer at 7:00 PM at Regal North screen 6",
        ]
        result = run_normal_pipeline(
            state=state,
            partner_text="Did you already take your medication?",
            camera_on=True,
            provided_face_signals={"face_detected": True, "confused_prob": 0.1, "negative_prob": 0.05, "nod_score": 0.0, "shake_score": 0.0},
            pb_enabled=True,
        )
        joined = " ".join(result.options).lower()
        self.assertIn("not sure", joined)
        self.assertTrue("8:30" in joined or "scheduled" in joined or "planned" in joined)
        self.assertNotIn("works for me", joined)
        self.assertNotIn("yes, that works for me", joined)
        self.assertNotIn("omer", joined)

    def test_medication_status_query_ignores_llm_candidate_rewrite(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        state.stm["today_plans"] = ["Take evening medication at 8:30 PM"]
        original_enabled = gemini_client.enabled
        original_generate = gemini_client.generate_candidates
        original_classify = gemini_client.classify_router
        original_refine = gemini_client.refine_evidence
        try:
            gemini_client.enabled = True
            gemini_client.classify_router = lambda text: ("Personal", "fake-router", "")
            gemini_client.refine_evidence = lambda text, evidence: (["bad llm rewrite"], "fake-refiner", "")
            gemini_client.generate_candidates = lambda **kwargs: (["Yes, that works for me.", "I remember Omer.", "Done."], "fake-generator", "")
            result = run_normal_pipeline(
                state=state,
                partner_text="Did you already take your medication?",
                camera_on=True,
                provided_face_signals={"face_detected": True, "confused_prob": 0.0, "negative_prob": 0.0, "nod_score": 0.0, "shake_score": 0.0},
                pb_enabled=True,
            )
            joined = " ".join(result.options).lower()
            self.assertIn("not sure", joined)
            self.assertNotIn("works for me", joined)
            self.assertNotIn("omer", joined)
        finally:
            gemini_client.enabled = original_enabled
            gemini_client.generate_candidates = original_generate
            gemini_client.classify_router = original_classify
            gemini_client.refine_evidence = original_refine

    def test_hi_how_are_you_is_treated_as_greeting_and_not_partner_profile(self):
        session = start_session("demo_user")
        state = memory_store.get_session(session["session_id"])
        state.stm["active_partner"] = {"person_id": "omer", "name": "Omer", "relation": "friend"}
        result = run_normal_pipeline(
            state=state,
            partner_text="Hi. How are you?",
            camera_on=True,
            provided_face_signals={"face_detected": True, "confused_prob": 0.6, "negative_prob": 0.05, "nod_score": 0.0, "shake_score": 0.0, "hand_gesture_label": "Open_Palm", "hand_gesture_score": 0.9},
            pb_enabled=True,
        )
        joined = " ".join(result.options).lower()
        self.assertNotIn("partner is", joined)
        self.assertNotIn("omer prefers", joined)
        self.assertTrue(any("hi" in option.lower() or "hey" in option.lower() or "hello" in option.lower() for option in result.options))
