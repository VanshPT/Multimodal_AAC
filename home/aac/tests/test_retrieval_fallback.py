from django.test import SimpleTestCase

from home.aac.memory.store import memory_store
from home.aac.pipelines.nodes import retrieve_from_pool_node


class RetrievalFallbackTests(SimpleTestCase):
    def test_retrieval_uses_fallback_when_needed(self):
        state = memory_store.start_session("demo_user")
        evidence, notes, trace = retrieve_from_pool_node(
            state=state,
            partner_text="Can we review tomorrow schedule and groceries?",
            search_order=["STM", "LTM", "PB"],
            chosen_buckets=["plans", "routine"],
            top_k=3,
        )
        pools = {item.pool for item in evidence}
        self.assertTrue(trace[0].pool == "STM")
        self.assertTrue(len(pools) >= 1)
        self.assertTrue(any("fallback" in note.lower() or "partial" in note.lower() for note in notes))
        self.assertTrue(len(trace) >= 1)
