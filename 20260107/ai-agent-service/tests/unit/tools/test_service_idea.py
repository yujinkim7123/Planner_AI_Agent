import unittest
import os
from dotenv import load_dotenv

from tools.service_idea_tools import create_service_ideas_tool, modify_service_ideas_tool
from validators.service_idea_validator import validate_service_ideas

@unittest.skipIf(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    "통합 테스트는 RUN_INTEGRATION_TESTS=true 환경 변수가 설정된 경우에만 실행됩니다."
)
class TestServiceIdeaToolsIntegration(unittest.TestCase):
    

    @classmethod
    def setUpClass(cls):
        
        load_dotenv()

    def setUp(self):
       
        self.persona = {
            "name": "박서준",
            "role": "꼼꼼한 위생관리맘",
            "demographics": "30대 후반, 맞벌이, 7세 아이 엄마",
            "pain_points": ["살균 기능의 실제 효과를 눈으로 확인할 수 없어 불안하다", "매번 옷을 삶는 것은 번거롭고 옷감이 상할까 걱정된다"]
        }
        self.cx_insights = {
            "opportunity_scores": [
                {"topic_id": "0-1", "action_keywords": ["살균", "세균", "바이러스"], "opportunity_score": 18.5},
                {"topic_id": "0-2", "action_keywords": ["시간", "단축", "자동"], "opportunity_score": 15.2}
            ]
        }
        self.product_context = {
            "product_name": "LG 트롬 세탁기",
            "features": ["스팀 살균", "AI 맞춤 세탁"],
            "sensors": ["의류 무게 감지 센서", "오염도 감지 센서"]
        }
        self.existing_ideas = [{
            "service_name": "AI 육아 위생 컨설턴트",
            "description": "페르소나의 아이 연령과 건강 상태에 맞춰, 의류, 장난감 등의 최적 살균 주기를 알려주는 서비스입니다.",
            "solved_pain_points": ["살균 기능의 실제 효과를 눈으로 확인할 수 없어 불안하다"],
            "service_scalability": "초기에는 ThinQ 앱 기능으로 제공하고, 추후 영유아 건강 데이터를 연동한 프리미엄 모델로 확장합니다."
        }]

    def test_create_service_ideas_tool_integration(self):
     
        print("\nRunning test: test_create_service_ideas_tool_integration")
        
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY 환경 변수가 설정되지 않아 통합 테스트를 건너뜁니다.")
        
        num_ideas = 2

        
        result = create_service_ideas_tool(
            persona=self.persona,
            cx_insights=self.cx_insights,
            product_context=self.product_context,
            num_ideas=num_ideas
        )

        
        print(f"\n[통합 테스트] LLM이 생성한 서비스 아이디어:\n{result}")

        self.assertIsNotNone(result)
        self.assertIn("service_ideas", result)
        
        validated_ideas = validate_service_ideas(result["service_ideas"])
        self.assertIsNotNone(validated_ideas, "LLM이 생성한 서비스 아이디어가 유효성 검사를 통과하지 못했습니다.")
        self.assertEqual(len(validated_ideas), num_ideas, f"{num_ideas}개의 아이디어가 생성되지 않았습니다.")

    def test_modify_service_ideas_tool_integration(self):

        print("\nRunning test: test_modify_service_ideas_tool_integration")
        
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY 환경 변수가 설정되지 않아 통합 테스트를 건너뜁니다.")

        modification_request = "기존 아이디어에 '구독형 모델'이라는 점을 명확히 하고, '가전 자동 제어' 기능을 상세 설명에 추가해주세요."
        
        
        result = modify_service_ideas_tool(
            existing_ideas=self.existing_ideas,
            modification_request=modification_request,
            persona=self.persona,
            cx_insights=self.cx_insights
        )

       
        print(f"\n[통합 테스트] LLM이 수정한 서비스 아이디어:\n{result}")

        self.assertIsNotNone(result)
        self.assertIn("service_ideas", result)

        validated_ideas = validate_service_ideas(result["service_ideas"])
        self.assertIsNotNone(validated_ideas, "LLM이 수정한 서비스 아이디어가 유효성 검사를 통과하지 못했습니다.")
     
        self.assertIn("구독형", str(validated_ideas[0]['description']), "수정 요청 사항('구독형')이 결과에 반영되지 않았습니다.")


if __name__ == '__main__':
    unittest.main()