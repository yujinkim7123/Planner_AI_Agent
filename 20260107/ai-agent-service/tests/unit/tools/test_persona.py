import unittest
import os
from dotenv import load_dotenv

from tools.persona_tools import create_personas_tool, modify_personas_tool
from validators.persona_validator import validate_personas

@unittest.skipIf(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    "통합 테스트는 RUN_INTEGRATION_TESTS=true 환경 변수가 설정된 경우에만 실행됩니다."
)
class TestPersonaToolsIntegration(unittest.TestCase):
    """
    실제 LLM API를 호출하여 페르소나 생성 및 수정 도구의 전체 흐름을 검증합니다.
    """

    @classmethod
    def setUpClass(cls):
        """테스트 클래스가 시작될 때 한 번만 .env 파일을 로드합니다."""
        load_dotenv()

    def setUp(self):
        """모든 테스트에서 공통으로 사용될 샘플 데이터를 준비합니다."""
      
        self.analysis_artifacts = {
            "topics_summary_list": [
                {"topic_id": "0-1", "action_keywords": ["가격", "가성비", "할인", "비싸다"]},
                {"topic_id": "0-2", "action_keywords": ["디자인", "색상", "UI", "예쁘다"]},
                {"topic_id": "0-3", "action_keywords": ["배송", "느리다", "파손", "포장"]}
            ]
        }
        self.web_results_sample = [
            {"original_text": "가격은 합리적인데 배송이 너무 느려서 실망했어요."},
            {"original_text": "디자인 하나만 보고 샀는데, 비싼 값을 하네요. 정말 예뻐요."},
            {"original_text": "가성비가 최고입니다. 이 가격에 이런 디자인이라니!"}
        ]
        self.user_request = "20대 대학생과 40대 직장인 페르소나를 만들어주세요."
        
       
        self.existing_personas = [{
            "name": "김민준",
            "role": "가성비를 중시하는 대학생",
            "demographics": "24세, 남성, 대학생",
            "behavioral_traits": ["쿠폰과 할인 정보를 꼼꼼히 챙김"],
            "needs_and_goals": ["제한된 예산 내에서 최대의 만족을 얻고 싶다"],
            "pain_points": ["디자인이 마음에 들면 너무 비싸다"],
            "motivating_quote": "같은 물건이라도 남들보다 싸게 사야 직성이 풀려요."
        }]

    def test_create_personas_tool_integration(self):
        """[통합 테스트] 실제 LLM을 호출하여 생성된 페르소나가 validator를 통과하는지 테스트합니다."""
        print("\nRunning test: test_create_personas_tool_integration")
        
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY 환경 변수가 설정되지 않아 통합 테스트를 건너뜁니다.")
        
        num_personas = 2

    
        result = create_personas_tool(
            analysis_artifacts=self.analysis_artifacts,
            web_results_sample=self.web_results_sample,
            num_personas=num_personas,
            user_request=self.user_request
        )

        
        print(f"\n[통합 테스트] LLM이 생성한 페르소나:\n{result}")

        self.assertIsNotNone(result)
        self.assertIn("personas", result)
        
        validated_personas = validate_personas(result["personas"])
        self.assertIsNotNone(validated_personas, "LLM이 생성한 페르소나가 유효성 검사를 통과하지 못했습니다.")
        self.assertEqual(len(validated_personas), num_personas, f"{num_personas}개의 페르소나가 생성되지 않았습니다.")

    def test_modify_personas_tool_integration(self):
        """[통합 테스트] 실제 LLM을 호출하여 수정된 페르소나가 validator를 통과하는지 테스트합니다."""
        print("\nRunning test: test_modify_personas_tool_integration")
        
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY 환경 변수가 설정되지 않아 통합 테스트를 건너뜁니다.")

        modification_request = "김민준 페르소나의 불편함(pain_points)에 '배송이 느린 점'을 추가해주세요."
        
     
        result = modify_personas_tool(
            existing_personas=self.existing_personas,
            modification_request=modification_request,
            analysis_artifacts=self.analysis_artifacts,
            web_results_sample=self.web_results_sample
        )

      
        print(f"\n[통합 테스트] LLM이 수정한 페르소나:\n{result}")

        self.assertIsNotNone(result)
        self.assertIn("personas", result)

        validated_personas = validate_personas(result["personas"])
        self.assertIsNotNone(validated_personas, "LLM이 수정한 페르소나가 유효성 검사를 통과하지 못했습니다.")
        
        self.assertIn("배송", str(validated_personas[0]['pain_points']), "수정 요청 사항이 결과에 반영되지 않았습니다.")


if __name__ == '__main__':
    unittest.main()