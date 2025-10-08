# tests/integration/agents/test_cx_analyst_agent.py

import unittest
import os
from dotenv import load_dotenv

from agents.experts.Analyst.cx_analyst_agent import run_cx_analyst_agent

from agents.common.graph_state import AgentState


load_dotenv()

RUN_E2E_TESTS = os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"

@unittest.skipUnless(RUN_E2E_TESTS, "Skipping End-to-End test. Set RUN_E2E_TESTS=true to run.")
class TestCXAnalystAgentE2E(unittest.TestCase):
 
    def setUp(self):
        """Initializes a realistic base state for the test."""
        self.initial_state: AgentState = {
            "project_id": 202,
            "user_request": "우리 제품 리뷰를 분석해서 고객들이 주로 뭘 원하는지, 개선할 기회가 뭔지 찾아줘.",
            "retrieved_data_summary": {
                "document_count": 50,
                "top_documents_sample": [
                    {"original_text": "카메라는 정말 좋지만, 배터리가 너무 빨리 닳아서 문제입니다.", "sentence_nouns": "카메라 정말 배터리 너무 문제"},
                    {"original_text": "배터리 수명만 개선된다면 최고의 핸드폰일 것입니다.", "sentence_nouns": "배터리 수명 개선 최고 핸드폰"},
                    {"original_text": "디자인은 예쁜데, 하루를 못 가는 배터리가 아쉬워요.", "sentence_nouns": "디자인 하루 배터리"},
                    {"original_text": "전반적으로 만족하지만 배터리가 조금 더 오래 갔으면 좋겠습니다.", "sentence_nouns": "전반적 만족 배터리 조금 더"},
                    {"original_text": "최고의 카메라. 야간 사진도 선명하게 잘 나옵니다.", "sentence_nouns": "최고 카메라 야간 사진 선명"},
                    {"original_text": "가격 대비 카메라 성능이 매우 뛰어납니다.", "sentence_nouns": "가격 대비 카메라 성능 매우"},
                    {"original_text": "성능이나 카메라 모두 만족하는데, 배터리가 문제입니다.", "sentence_nouns": "성능 카메라 모두 만족 배터리 문제"},
                ]
            },
            "analysis_options": {
                "num_clusters": 2, 
                "num_topics_per_cluster": 2,
                "target_cluster_id_for_sna": 0,
                "target_cluster_id_for_lda": 0
            },
            "cx_insights": {}, "topics": None, "insights_summary": None, "personas": None,
            "service_ideas": None, "data_plan": None, "final_document": None,
            "current_observation": "Project started.", "next_action": None
        }
        print("\n--- Starting E2E Test for CX Analyst Agent ---")

    def test_end_to_end_execution(self):
        """
        Scenario: Runs the entire agent workflow from planning to final summary.
        It verifies that the agent completes successfully and the final state is valid.
        """
        
        result = run_cx_analyst_agent(self.initial_state)

       
        self.assertEqual(result.get("next_action"), "success", f"Agent failed with reason: {result.get('reason')}")

        
        final_state = result.get("updated_state", {})
        self.assertIsNotNone(final_state.get("cx_insights"), "cx_insights should be populated.")

        cx_insights = final_state["cx_insights"]
        print("\n--- Final CX Insights for Verification ---")
        print(cx_insights)
        print("----------------------------------------")

        
        self.assertIn("clustering", cx_insights)
        self.assertIn("lda", cx_insights)
        
        if "sna" in cx_insights:
            self.assertIn("graph_data", cx_insights["sna"])
        self.assertIn("scores", cx_insights)
        self.assertIn("summary", cx_insights)

       
        topics = result.get("topics")
        self.assertIsInstance(topics, list)
        self.assertGreater(len(topics), 0, "'topics' list should not be empty.")
        self.assertIsInstance(topics[0], dict)
        self.assertIn("topic_id", topics[0])
        self.assertIn("opportunity_score", topics[0])

       
        summary = cx_insights.get("summary")
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 10, "Summary string should not be trivial.")

        print("\n--- E2E Test for CX Analyst Agent PASSED ---")

if __name__ == '__main__':
    unittest.main()