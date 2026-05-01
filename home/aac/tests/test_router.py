from django.test import SimpleTestCase

from home.aac.pipelines.nodes import _is_binary_prompt, router_node, source_priority_planner_node


class RouterNodeTests(SimpleTestCase):
    def test_router_personal(self):
        label = router_node("What is your favorite food?")
        self.assertEqual(label, "Personal")

    def test_router_contextual(self):
        label = router_node("Are we still on for tonight?")
        self.assertEqual(label, "Contextual")

    def test_router_open_domain(self):
        label = router_node("What is reinforcement learning?")
        self.assertEqual(label, "Open-domain")

    def test_planner_order(self):
        self.assertEqual(source_priority_planner_node("Contextual"), ["STM", "LTM", "PB"])

    def test_binary_prompt_detection(self):
        self.assertTrue(_is_binary_prompt("Do you want a reminder for medication?"))
        self.assertTrue(_is_binary_prompt("Are we still on for 7?"))
        self.assertFalse(_is_binary_prompt("Tell me your favorite food."))
