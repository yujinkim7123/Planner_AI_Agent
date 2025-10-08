import unittest
import os

from tools.nlp.analysis_summary import run

@unittest.skipIf(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    "통합 테스트는 RUN_INTEG`RATION_TESTS=true 환경 변수가 설정된 경우에만 실행됩니다."
)
class TestAnalysisSummaryIntegration(unittest.TestCase):

    def setUp(self):
        """테스트에 사용될 샘플 토픽 데이터를 준비합니다."""
        self.sample_topics = [
            {
                "topic_id": "0-1", 
                "action_keywords": ["가격", "오류", "결제", "비싸다"], 
                "opportunity_score": 18.5,
                "importance": 9.8,
                "satisfaction": 1.3
            },
            {
                "topic_id": "0-2", 
                "action_keywords": ["디자인", "색상", "UI", "예쁘다"], 
                "opportunity_score": 12.3,
                "importance": 7.5,
                "satisfaction": 5.2
            },
        ]

    def test_run_integration_end_to_end(self):
   
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY 환경 변수가 설정되지 않아 통합 테스트를 건너뜁니다.")

        result = run(self.sample_topics)


        self.assertIsNotNone(result, "LLM 응답이 None입니다.")
        self.assertIsInstance(result, str, "LLM 응답이 문자열(string) 형태가 아닙니다.")
        self.assertGreater(len(result), 20, "LLM 응답이 너무 짧거나 비어있습니다.") # 최소 길이를 20자로 가정
        self.assertNotIn("오류가 발생했습니다", result, "LLM 호출 중 에러 메시지가 반환되었습니다.")
                

if __name__ == '__main__':
    unittest.main()