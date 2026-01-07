import unittest
import os
from dotenv import load_dotenv

# 실제 프로젝트 구조에 맞게 경로를 조정해주세요.
from tools.data_plan_tools import create_data_plan_tool, modify_data_plan_tool
from validators.data_plan_validator import validate_data_plan

@unittest.skipIf(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    "통합 테스트는 RUN_INTEGRATION_TESTS=true 환경 변수가 설정된 경우에만 실행됩니다."
)
class TestDataPlanToolsIntegration(unittest.TestCase):
    """
    실제 LLM API를 호출하여 데이터 기획안 생성 및 수정 도구의 전체 흐름을 검증합니다.
    """

    @classmethod
    def setUpClass(cls):
        """테스트 클래스가 시작될 때 한 번만 .env 파일을 로드합니다."""
        load_dotenv()

    def setUp(self):
        """모든 테스트에서 공통으로 사용될 샘플 데이터를 준비합니다."""
        self.service_idea = {
            "name": "AI 기반 맞춤형 여행 코스 추천 서비스",
            "description": "사용자의 취향, 예산, 여행 기간을 입력하면 AI가 최적의 여행 코스와 맛집, 숙소를 추천해주는 모바일 앱 서비스."
        }
        self.product_context = {
            "target_audience": "자유여행을 선호하는 20-30대 MZ세대",
            "key_features": ["개인화 추천 엔진", "실시간 예약 연동", "사용자 리뷰 기반 평점 시스템"]
        }
        self.existing_plan = {
            "service_name": "AI 여행 플래너",
            "data_driven_features": [
                {
                    "idea_name": "숨은 맛집 추천 고도화",
                    "description": "사용자의 과거 리뷰 데이터와 실시간 위치를 분석하여, 대중적이지는 않지만 평점이 높은 현지인 맛집을 추천합니다.",
                    "required_data": ["사용자 리뷰 텍스트", "실시간 GPS 데이터", "식당 평점 데이터"]
                }
            ],
            "inferred_insights": [
                {
                    "idea_name": "여행지 혼잡도 예측",
                    "description": "과거 시간대별 사용자 위치 데이터와 공휴일 정보를 결합하여, 특정 관광지의 미래 혼잡도를 예측하고 사용자에게 최적의 방문 시간을 제안합니다.",
                    "required_sensors": ["GPS 센서 데이터", "달력 API 연동 데이터"]
                }
            ],
            "new_data_sources": [
                {
                    "source_type": "외부 데이터 제휴",
                    "source_name": "지역별 날씨 API",
                    "collectable_data": "여행지의 실시간 및 주간 날씨 예보 데이터 (기온, 강수확률 등)",
                    "value_proposition": "날씨 변화에 따른 대체 여행 코스를 실시간으로 추천하여 사용자의 여행 만족도를 높입니다."
                }
            ]
        }
    def test_create_data_plan_tool_integration(self):
        """[통합 테스트] 실제 LLM을 호출하여 생성된 데이터 기획안이 validator를 통과하는지 테스트합니다."""
        print("\nRunning test: test_create_data_plan_tool_integration")
        
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY 환경 변수가 설정되지 않아 통합 테스트를 건너뜁니다.")

        # WHEN: Mock 없이 실제 생성 도구를 실행
        result = create_data_plan_tool(
            service_idea=self.service_idea,
            product_context=self.product_context
        )

        # THEN: 반환된 결과의 'data_plan' 부분이 validator의 검증을 통과해야 함
        print(f"\n[통합 테스트] LLM이 생성한 데이터 기획안:\n{result}")
        
        self.assertIsNotNone(result)
        self.assertIn("data_plan", result)
        
        validated_plan = validate_data_plan(result["data_plan"])
        self.assertIsNotNone(validated_plan, "LLM이 생성한 데이터 기획안이 유효성 검사를 통과하지 못했습니다.")

    def test_modify_data_plan_tool_integration(self):
        """[통합 테스트] 실제 LLM을 호출하여 수정된 데이터 기획안이 validator를 통과하는지 테스트합니다."""
        print("\nRunning test: test_modify_data_plan_tool_integration")

        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY 환경 변수가 설정되지 않아 통합 테스트를 건너뜁니다.")

        modification_request = "사용자 만족도 지표(CSAT)를 추적하기 위한 데이터 소스를 추가해 주세요."

        # WHEN: Mock 없이 실제 수정 도구를 실행
        result = modify_data_plan_tool(
            existing_plan=self.existing_plan,
            modification_request=modification_request,
            service_idea=self.service_idea
        )

        # THEN: 반환된 결과의 'data_plan' 부분이 validator의 검증을 통과해야 함
        print(f"\n[통합 테스트] LLM이 수정한 데이터 기획안:\n{result}")

        self.assertIsNotNone(result)
        self.assertIn("data_plan", result)

        validated_plan = validate_data_plan(result["data_plan"])
        self.assertIsNotNone(validated_plan, "LLM이 수정한 데이터 기획안이 유효성 검사를 통과하지 못했습니다.")


if __name__ == '__main__':
    unittest.main()