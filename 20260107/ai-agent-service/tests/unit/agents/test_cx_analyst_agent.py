# tests/integration/agents/test_cx_analyst_agent.py

import unittest
from unittest.mock import patch, MagicMock
from copy import deepcopy

from agents.experts.Analyst.cx_analyst_agent import run_cx_analyst_agent
from agents.common.graph_state import AgentState

class TestCXAnalystAgent(unittest.TestCase):
    def setUp(self):
        self.base_state: AgentState = {
            "project_id": "test_project", "user_request": "데이터를 분석해줘",
            "retrieved_data_summary": {}, "analysis_options": {}, "product_type": "test_product",
            "cx_insights": None, "insights_summary": None, "topics": None, "personas": None,
            "service_ideas": None, "data_plan": None, "final_document": None,
            "current_observation": "Project started.", "next_action": None
        }

    @patch('agents.experts.Analyst.cx_analyst_agent.registry')
    @patch('agents.experts.Analyst.cx_analyst_agent.planner')
    def test_full_end_to_end_successful_workflow(self, mock_planner, mock_registry):
        """(시나리오) 모든 분석 도구를 순차적으로 실행하는 전체 성공 워크플로우를 테스트합니다."""
        initial_state = deepcopy(self.base_state)
        
        # 1. Planner가 모든 분석 단계를 포함하는 완벽한 계획을 반환하도록 설정
        mock_planner.create_plan_list.return_value = {
            "plan_list": [
                {"action": "run_clustering"},
                {"action": "run_lda"},
                {"action": "run_sna"},
                {"action": "calculate_opportunity_scores"},
                {"action": "finish"}
            ]
        }

        # 2. 각 도구에 대한 Mock 객체를 생성
        mock_clustering_tool = MagicMock(return_value={"status": "Clustering complete"})
        mock_lda_tool = MagicMock(return_value={"status": "LDA complete"})
        mock_sna_tool = MagicMock(return_value={"status": "SNA complete"})
        mock_scores_tool = MagicMock(return_value=[{"topic_id": "0-1", "opportunity_score": 15.0}])
        mock_summary_tool = MagicMock(return_value="모든 분석이 완료된 최종 요약문입니다.")

        # 3. Registry가 각 도구 이름에 맞는 Mock을 정확히 반환하도록 설정
        def get_task_info_side_effect(tool_name):
            if tool_name == "run_clustering":
                return {"tool": mock_clustering_tool, "validator": lambda x: x, "params_builder": lambda s: {}, "payload_key": "clustering"}
            elif tool_name == "run_lda":
                return {"tool": mock_lda_tool, "validator": lambda x: x, "params_builder": lambda s: {}, "payload_key": "lda"}
            elif tool_name == "run_sna":
                return {"tool": mock_sna_tool, "validator": lambda x: x, "params_builder": lambda s: {}, "payload_key": "sna"}
            elif tool_name == "calculate_opportunity_scores":
                return {"tool": mock_scores_tool, "validator": lambda x: x, "params_builder": lambda s: {}, "payload_key": "scores"}
            elif tool_name == "create_summary":
                return {"tool": mock_summary_tool, "validator": lambda x: x, "params_builder": lambda s: {}, "payload_key": "summary"}
            return None
        
        mock_registry.get_task_info.side_effect = get_task_info_side_effect

        # --- 에이전트 실행 ---
        result = run_cx_analyst_agent(initial_state)

        # 4. 최종 결과가 'success'인지 확인
        self.assertEqual(result.get("next_action"), "success")
        
        # 5. 모든 Mock 도구가 정확히 한 번씩 호출되었는지 확인
        mock_clustering_tool.assert_called_once()
        mock_lda_tool.assert_called_once()
        mock_sna_tool.assert_called_once()
        mock_scores_tool.assert_called_once()
        mock_summary_tool.assert_called_once()

        # 6. 최종 상태에 모든 분석 결과가 잘 저장되었는지 확인
        final_state = result['updated_state']
        self.assertIn("clustering", final_state['cx_insights'])
        self.assertIn("lda", final_state['cx_insights'])
        self.assertIn("sna", final_state['cx_insights'])
        self.assertIn("scores", final_state['cx_insights'])
        self.assertIn("summary", final_state['cx_insights'])
        self.assertEqual(final_state['cx_insights']['summary'], "모든 분석이 완료된 최종 요약문입니다.")
        

    @patch('agents.experts.Analyst.cx_analyst_agent.registry')
    @patch('agents.experts.Analyst.cx_analyst_agent.planner')
    def test_tool_execution_failure(self, mock_planner, mock_registry):
        initial_state = deepcopy(self.base_state)
        
        mock_planner.create_plan_list.return_value = {
            "plan_list": [{"action": "run_clustering"}]
        }
        
        mock_failing_tool = MagicMock(side_effect=Exception("Tool failed!"))
        mock_registry.get_task_info.return_value = {
            "tool": mock_failing_tool, "validator": lambda x: x,
            "params_builder": lambda s: {}, "payload_key": "clustering_result"
        }

        result = run_cx_analyst_agent(initial_state)

        self.assertEqual(result.get("next_action"), "error")
        self.assertIn("Tool failed!", result.get("reason", ""))
        mock_failing_tool.assert_called_once()