import unittest
import os
from dotenv import load_dotenv

from tools.final_document_tools import create_final_document_tool, modify_final_document_tool
from validators.final_document_validator import validate_final_document

@unittest.skipIf(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    "통합 테스트는 RUN_INTEGRATION_TESTS=true 환경 변수가 설정된 경우에만 실행됩니다."
)
class TestFinalDocumentToolsIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_dotenv()

    def setUp(self):

        self.persona = {
            "name": "박서준", "role": "꼼꼼한 위생관리맘", "demographics": "30대 후반, 맞벌이, 7세 아이 엄마",
            "motivating_quote": "아이가 쓰는 건데, 조금 비싸더라도 확실한 걸로 사야 마음이 놓여요."
        }
        self.service_idea = {
            "service_name": "AI 육아 위생 컨설턴트",
            "description": "아이 연령과 건강 상태에 맞춰 최적 살균 주기를 알려주고 가전을 자동으로 제어하는 구독형 서비스",
            "solved_pain_points": ["살균 기능의 실제 효과를 눈으로 확인할 수 없어 불안하다", "매번 옷을 삶는 것은 번거롭다"],
            "service_scalability": "초기에는 ThinQ 앱 기능으로 제공, 추후 영유아 건강 데이터를 연동한 프리미엄 유료 모델로 확장"
        }
        self.data_plan = {
            "service_name": "AI 육아 위생 컨설턴트",
            "data_driven_features": [{"idea_name": "살균 주기 개인화 추천", "description": "...", "required_data": ["사용자 가전 사용 로그"]}],
            "inferred_insights": [{"idea_name": "가족 건강 이상 징후 감지", "description": "...", "required_sensors": ["실내 먼지 센서"]}],
            "new_data_sources": [{"source_type": "신규 센서", "source_name": "식기 오염도 센서", "collectable_data": "...", "value_proposition": "..."}]
        }
        self.existing_document = {
            "title": "유첨. AI 육아 위생 컨설턴트 최종 보고서",
            "customer_delight_goal": "데이터로 증명하는 안심, 스마트한 육아의 시작",
            "cx": {"target_definition": {"description": "...", "quote": "...", "market_info": "..."}, "core_experience": {"title": "...", "care": "...", "customization": [], "servitization": "..."}},
            "performance": {"concept": {"find": "...", "unique": []}},
            "dx": {"trigger": {"title": "...", "items": []}, "accelerator": {"title": "...", "up_contents_service": [], "data_driven_experience": []}, "tracker": {"title": "...", "items": []}}
        }

    def test_create_final_document_tool_integration(self):
        
        print("\nRunning test: test_create_final_document_tool_integration")
        
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY 환경 변수가 설정되지 않아 통합 테스트를 건너뜁니다.")
   
        result = create_final_document_tool(
            persona=self.persona,
            service_idea=self.service_idea,
            data_plan=self.data_plan
        )

        
        print(f"\n[통합 테스트] LLM이 생성한 최종 보고서:\n{result}")

        self.assertIsNotNone(result)
        self.assertIn("final_document", result)
        
        validated_doc = validate_final_document(result["final_document"])
        self.assertIsNotNone(validated_doc, "LLM이 생성한 최종 보고서가 유효성 검사를 통과하지 못했습니다.")
        self.assertEqual(validated_doc['cx']['target_definition']['quote'], self.persona['motivating_quote'])

    def test_modify_final_document_tool_integration(self):
        
        print("\nRunning test: test_modify_final_document_tool_integration")
        
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY 환경 변수가 설정되지 않아 통합 테스트를 건너뜁니다.")

        modification_request = "고객 감동 목표(customer_delight_goal)를 좀 더 간결하고 임팩트 있는 문장으로 바꿔주세요."
        
       
        result = modify_final_document_tool(
            existing_document=self.existing_document,
            modification_request=modification_request,
            persona=self.persona,
            service_idea=self.service_idea,
            data_plan=self.data_plan
        )

        
        print(f"\n[통합 테스트] LLM이 수정한 최종 보고서:\n{result}")

        self.assertIsNotNone(result)
        self.assertIn("final_document", result)

        validated_doc = validate_final_document(result["final_document"])
        self.assertIsNotNone(validated_doc, "LLM이 수정한 최종 보고서가 유효성 검사를 통과하지 못했습니다.")
       
        self.assertNotEqual(validated_doc['customer_delight_goal'], self.existing_document['customer_delight_goal'])


if __name__ == '__main__':
    unittest.main()