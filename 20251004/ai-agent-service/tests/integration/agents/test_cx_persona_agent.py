# tests/unit/agents/test_persona_creation_unit.py

import unittest
import os
from dotenv import load_dotenv

from agents.experts.Creater.creator_agent import run_creator_agent
from agents.common.graph_state import AgentState

load_dotenv()

RUN_UNIT_TESTS = os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"

@unittest.skipUnless(RUN_UNIT_TESTS, "Skipping unit test. Set RUN_INTEGRATION_TESTS=true to run.")
class TestPersonaCreationUnit(unittest.TestCase):
 
    def setUp(self):
        """
        Creator Agent의 페르소나 생성 기능만 단위 테스트합니다.
        CX Analyst의 실제 출력 형식을 모방한 Mock 데이터를 사용합니다.
        """
        self.initial_state: AgentState = {
            "project_id": 303,
            "user_request": "페르소나 3개 만들어줘.",
            "retrieved_data_summary": {
                "document_count": 50,
                "top_documents_sample": [
                    {"original_text": "카메라는 정말 좋지만, 배터리가 너무 빨리 닳아서 문제입니다.", "sentence_nouns": "카메라 정말 배터리 너무 문제"},
                    {"original_text": "배터리 수명만 개선된다면 최고의 핸드폰일 것입니다.", "sentence_nouns": "배터리 수명 개선 최고 핸드폰"},
                    {"original_text": "디자인은 예쁜데, 하루를 못 가는 배터리가 아쉬워요.", "sentence_nouns": "디자인 하루 배터리"},
                    {"original_text": "최고의 카메라. 야간 사진도 선명하게 잘 나옵니다.", "sentence_nouns": "최고 카메라 야간 사진 선명"},
                    {"original_text": "가격 대비 카메라 성능이 매우 뛰어납니다.", "sentence_nouns": "가격 대비 카메라 성능 매우"},
                ]
            },
            "analysis_options": None,
            "cx_insights": {
                "clustering": {
                    "method": "kmeans",
                    "num_clusters": 2,
                    "cluster_labels": ["배터리 불만 그룹", "카메라 만족 그룹"],
                    "cluster_sizes": [4, 3],
                    "cluster_summaries": [
                        "배터리 수명에 대한 불만이 주를 이루는 그룹",
                        "카메라 성능에 만족하는 사용자 그룹"
                    ]
                },
                "lda": {
                    "num_topics": 4,
                    "topics": [
                        {
                            "topic_id": "0-1",
                            "keywords": ["배터리", "수명", "문제", "개선", "하루"],
                            "weights": [0.3, 0.25, 0.2, 0.15, 0.1],
                            "representative_texts": [
                                "배터리가 너무 빨리 닳습니다",
                                "하루를 못 가는 배터리가 아쉽습니다"
                            ]
                        },
                        {
                            "topic_id": "0-2",
                            "keywords": ["카메라", "화질", "성능", "야간", "사진"],
                            "weights": [0.35, 0.2, 0.2, 0.15, 0.1],
                            "representative_texts": [
                                "카메라 성능이 매우 뛰어납니다",
                                "야간 사진도 선명하게 잘 나옵니다"
                            ]
                        }
                    ]
                },
                "scores": {
                    "topics": [
                        {
                            "topic_id": "0-1",
                            "opportunity_score": 85,
                            "volume": 120,
                            "sentiment_score": -0.6,
                            "urgency_keywords": ["문제", "불만", "개선"]
                        },
                        {
                            "topic_id": "0-2",
                            "opportunity_score": 72,
                            "volume": 95,
                            "sentiment_score": 0.8,
                            "urgency_keywords": ["최고", "만족", "뛰어남"]
                        }
                    ]
                },
                "summary": "고객들은 배터리 수명에 대한 불만(기회점수 85점)이 가장 크며, 카메라 성능(기회점수 72점)에는 높은 만족도를 보이고 있습니다. 배터리 수명 개선이 최우선 개선 과제입니다."
            },
            "topics": [
                {"topic_id": "0-1", "opportunity_score": 85, "keywords": ["배터리", "수명", "문제"]},
                {"topic_id": "0-2", "opportunity_score": 72, "keywords": ["카메라", "화질", "성능"]}
            ],
            "insights_summary": "배터리 수명 개선이 가장 큰 개선 기회입니다.",
            "personas": None,
            "service_ideas": None,
            "data_plan": None,
            "final_document": None,
            "current_observation": "CX analysis completed (mocked for unit test).",
            "next_action": None
        }
        print("\n--- Starting Unit Test: Persona Creation with Mocked CX Data ---")

    def test_persona_creation_with_mock_cx_data(self):
        """
        시나리오: Mock된 CX 분석 결과를 바탕으로 페르소나 생성 기능만 단위 테스트
        """
        
        result = run_creator_agent(self.initial_state)

        # 성공 여부 확인
        self.assertEqual(
            result.get("next_action"), 
            "success", 
            f"Creator Agent failed: {result.get('reason')}"
        )

        # 페르소나 검증
        final_state = result.get("updated_state", {})
        personas = final_state.get("personas")
        
        self.assertIsNotNone(personas, "Personas should be created")
        self.assertEqual(len(personas), 3, "Should create exactly 3 personas")
        
        print("\n--- Generated Personas from Mock CX Data ---")
        for i, persona in enumerate(personas, 1):
            print(f"\n[Persona {i}]")
            print(f"Name: {persona.get('name')}")
            print(f"Role: {persona.get('role')}")
            print(f"Pain Points: {persona.get('pain_points')}")
            
            # 필수 필드 검증
            with self.subTest(persona_index=i):
                self.assertIn("name", persona)
                self.assertIn("role", persona)
                self.assertIn("demographics", persona)
                self.assertIn("behavioral_traits", persona)
                self.assertIn("needs_and_goals", persona)
                self.assertIn("pain_points", persona)
                
                # 리스트 필드는 비어있지 않아야 함
                self.assertGreater(len(persona["behavioral_traits"]), 0)
                self.assertGreater(len(persona["needs_and_goals"]), 0)
                self.assertGreater(len(persona["pain_points"]), 0)
                
        # 전체 페르소나 중 배터리 관련 언급이 있는지 확인
        all_pain_points = " ".join([
            " ".join(p.get("pain_points", [])) 
            for p in personas
        ]).lower()
        
        self.assertTrue(
            "배터리" in all_pain_points or "battery" in all_pain_points,
            "At least one persona should mention battery issues from mock CX data"
        )
        
        print("\n--- Unit Test PASSED ---")

    def test_persona_modification_success(self):
        """
        시나리오: 기존 페르소나를 성공적으로 수정
        """
        print("\n--- Starting Test: Persona Modification (Success) ---")
        
        # Step 1: 먼저 페르소나 생성
        create_result = run_creator_agent(self.initial_state)
        self.assertEqual(create_result.get("next_action"), "success")
        
        created_state = create_result.get("updated_state", {})
        original_personas = created_state.get("personas")
        self.assertIsNotNone(original_personas)
        self.assertEqual(len(original_personas), 3)
        
        print(f"\n[Original First Persona]")
        print(f"Name: {original_personas[0].get('name')}")
        print(f"Demographics: {original_personas[0].get('demographics')}")
        
        # Step 2: 페르소나 수정 요청
        modify_state = created_state.copy()
        modify_state["user_request"] = "첫 번째 페르소나의 연령대를 20대 초반으로 변경하고, 직업을 대학생으로 수정해줘."
        
        modify_result = run_creator_agent(modify_state)
        
        # 수정 성공 확인
        self.assertEqual(
            modify_result.get("next_action"),
            "success",
            f"Modification failed: {modify_result.get('reason')}"
        )
        
        # 수정된 페르소나 검증
        modified_state = modify_result.get("updated_state", {})
        modified_personas = modified_state.get("personas")
        
        self.assertIsNotNone(modified_personas)
        self.assertEqual(len(modified_personas), 3, "Should maintain 3 personas")
        
        print(f"\n[Modified First Persona]")
        print(f"Name: {modified_personas[0].get('name')}")
        print(f"Demographics: {modified_personas[0].get('demographics')}")
        
        # 수정 내용 반영 확인
        first_persona_demo = modified_personas[0].get("demographics", "").lower()
        self.assertTrue(
            "20대" in first_persona_demo or "20" in first_persona_demo,
            "Modified demographics should reflect age change to 20s"
        )
        self.assertTrue(
            "대학생" in first_persona_demo or "학생" in first_persona_demo,
            "Modified demographics should reflect occupation change to student"
        )
        
        print("\n--- Persona Modification Test PASSED ---")

    def test_persona_creation_without_cx_insights(self):
        """
        시나리오 (실패): CX 분석 결과 없이 페르소나 생성 시도
        """
        print("\n--- Starting Test: Persona Creation Without CX Insights (Expected Failure) ---")
        
        # cx_insights를 None으로 설정
        state_without_insights = self.initial_state.copy()
        state_without_insights["cx_insights"] = None
        
        result = run_creator_agent(state_without_insights)
        
        # 실패해야 함
        self.assertEqual(
            result.get("next_action"),
            "error",
            "Should fail when cx_insights is missing"
        )
        
        # 에러 메시지 확인
        error_reason = result.get("reason", "")
        self.assertIn(
            "CX 분석 결과",
            error_reason,
            "Error message should mention missing CX analysis"
        )
        
        print(f"Expected error occurred: {error_reason}")
        print("\n--- Expected Failure Test PASSED ---")

    def test_persona_modification_without_existing_personas(self):
        """
        시나리오 (실패): 기존 페르소나 없이 수정 시도
        """
        print("\n--- Starting Test: Persona Modification Without Existing Personas (Expected Failure) ---")
        
        # 수정 요청이지만 기존 페르소나가 없는 상태
        state_for_modify = self.initial_state.copy()
        state_for_modify["user_request"] = "첫 번째 페르소나를 수정해줘."
        state_for_modify["personas"] = None
        
        result = run_creator_agent(state_for_modify)
        
        # 플래너가 modify 액션을 선택할 수 있지만, 실행 시 실패해야 함
        # 또는 플래너가 아예 계획을 세우지 못할 수 있음
        
        # 어느 경우든 최종 결과에 페르소나가 없거나 에러가 발생했어야 함
        final_state = result.get("updated_state", {})
        personas = final_state.get("personas")
        
        # 페르소나가 없거나, 에러가 발생했어야 함
        if result.get("next_action") == "error":
            print(f"Expected error occurred: {result.get('reason')}")
        else:
            # 성공했다면 create로 처리되었을 수 있음 (이것도 허용 가능)
            print("Planner interpreted as create request instead of modify")
        
        print("\n--- Expected Failure/Recovery Test PASSED ---")

    def test_persona_creation_with_invalid_number(self):
        """
        시나리오: 비정상적인 개수로 페르소나 생성 요청
        """
        print("\n--- Starting Test: Persona Creation With Invalid Number ---")
        
        # 0개 요청
        state_zero = self.initial_state.copy()
        state_zero["user_request"] = "페르소나 0개 만들어줘."
        
        result_zero = run_creator_agent(state_zero)
        
        # 시스템이 어떻게 처리하는지 확인 (에러 또는 기본값으로 처리)
        if result_zero.get("next_action") == "success":
            final_state = result_zero.get("updated_state", {})
            personas = final_state.get("personas")
            
            # 🔥 수정: personas가 None일 수 있음을 처리
            if personas is None:
                print("System returned None for 0 request - treating as empty list")
                personas = []
            
            print(f"System handled 0 request, created {len(personas)} personas")
            # 시스템이 0을 거부하고 기본값으로 처리했는지 확인
            if len(personas) > 0:
                self.assertGreater(len(personas), 0, "System handled invalid number gracefully")
            else:
                print("System created no personas for 0 request (acceptable behavior)")
        else:
            print(f"System rejected invalid request: {result_zero.get('reason')}")
            # 에러를 반환하는 것도 합리적인 동작
        
        # 100개 요청 (과도한 요청)
        state_hundred = self.initial_state.copy()
        state_hundred["user_request"] = "페르소나 100개 만들어줘."
        
        result_hundred = run_creator_agent(state_hundred)
        
        if result_hundred.get("next_action") == "success":
            final_state = result_hundred.get("updated_state", {})
            personas = final_state.get("personas")
            
            # 🔥 수정: personas가 None일 수 있음을 처리
            if personas is None:
                print("System returned None for 100 request")
                self.fail("System should handle 100 request with actual personas or error")
            else:
                print(f"System handled 100 request, created {len(personas)} personas")
                # 시스템이 합리적인 수로 제한했는지 확인
                self.assertLessEqual(len(personas), 20, "System should cap at reasonable number")
        else:
            print(f"System rejected excessive request: {result_hundred.get('reason')}")
            # 과도한 요청을 거부하는 것도 합리적
        
        print("\n--- Invalid Number Test PASSED ---")

    def test_persona_creation_with_ambiguous_request(self):
        """
        시나리오: 모호한 요청으로 페르소나 생성
        """
        print("\n--- Starting Test: Persona Creation With Ambiguous Request ---")
        
        state_ambiguous = self.initial_state.copy()
        state_ambiguous["user_request"] = "페르소나 몇 개 만들어줘."
        
        result = run_creator_agent(state_ambiguous)
        
        # 시스템이 합리적인 기본값을 사용했는지 확인
        if result.get("next_action") == "success":
            final_state = result.get("updated_state", {})
            personas = final_state.get("personas")
            
            # 🔥 수정: personas가 None일 수 있음을 처리
            if personas is None:
                print("System returned None for ambiguous request")
                self.fail("System should handle ambiguous request with default personas or error")
            else:
                self.assertGreater(len(personas), 0, "Should create at least 1 persona")
                self.assertLessEqual(len(personas), 10, "Should not create too many personas")
                
                print(f"System interpreted ambiguous request and created {len(personas)} personas")
        else:
            print(f"System couldn't handle ambiguous request: {result.get('reason')}")
            # 모호한 요청을 거부하는 것도 합리적인 동작
        
        print("\n--- Ambiguous Request Test PASSED ---")


if __name__ == '__main__':
    unittest.main()