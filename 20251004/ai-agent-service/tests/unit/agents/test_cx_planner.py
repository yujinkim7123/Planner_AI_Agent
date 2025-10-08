# tests/integration/test_planner_integration.py

import unittest
import os
from dotenv import load_dotenv

# 실제 Planner를 테스트
from agents.experts.Analyst.planner import planner

# .env 파일에서 실제 API 키를 로드해야 합니다.
load_dotenv()

# CI/CD 환경이나 로컬에서 통합 테스트를 실행할지 여부를 결정하는 환경 변수
# 터미널에서 `set RUN_INTEGRATION_TESTS=true` 와 같이 설정 후 테스트 실행
RUN_INTEGRATION_TESTS = os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"

@unittest.skipUnless(RUN_INTEGRATION_TESTS, "실제 LLM 호출이 필요한 통합 테스트는 기본적으로 건너뜁니다.")
class TestPlannerIntegration(unittest.TestCase):
    """실제 LLM을 호출하여 Planner의 통합 기능을 검증합니다."""
    
    def test_planner_with_real_llm_for_basic_request(self):
        """(시나리오) 기본적인 분석 요청에 대해 유효한 구조의 계획을 생성하는지 테스트합니다."""
     
        user_request = "수집된 고객 리뷰 데이터를 클러스터링하고, 각 그룹별로 핵심 토픽을 뽑아서 기회 점수를 계산해줘."
    
        main_state = {
            "retrieved_data_summary": {
                "document_count": 150,
                "top_documents_sample": [
                    {"original_text": "디자인은 좋은데 배터리가 너무 빨리 닳아요."},
                    {"original_text": "배터리 문제만 해결되면 완벽한 제품입니다."},
                    {"original_text": "카메라 성능이 정말 마음에 듭니다."}
                ]
            },
           
            "cx_insights": {} 
        }

      
        result = planner.create_plan_list(user_request, main_state)
        
       
        print("\n[실제 LLM이 생성한 계획]:", result.get("plan_list"))

     
        self.assertNotIn("error_message", result, f"Planner 실행 중 오류 발생: {result.get('error_message')}")
        
       
        self.assertIn("plan_list", result)
        plan_list = result["plan_list"]
        
      
        self.assertIsInstance(plan_list, list)
        self.assertTrue(len(plan_list) > 0, "LLM이 유효한 계획을 생성하지 못했습니다.")
        
        
        first_step = plan_list[0]
        self.assertIsInstance(first_step, dict)
        self.assertIn("action", first_step)
        
    
        action_sequence = [step['action'] for step in plan_list]
        if "run_lda" in action_sequence:
            self.assertIn("run_clustering", action_sequence, "'run_lda'가 계획에 있으나 'run_clustering'이 없습니다.")
            self.assertLess(
                action_sequence.index("run_clustering"), 
                action_sequence.index("run_lda"),
                "'run_clustering'은 'run_lda'보다 먼저 실행되어야 합니다."
            )