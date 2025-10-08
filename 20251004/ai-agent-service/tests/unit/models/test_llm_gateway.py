import unittest
import os

# .env 파일 로드를 위해 추가 (테스트 실행의 시작점이므로)
from dotenv import load_dotenv

# 실제 프로젝트 구조에 맞게 경로를 조정해주세요.
from models.llm_gateway import call_llm

# 이 데코레이터는 통합 테스트를 실수로 실행하는 것을 방지하는 안전장치입니다.
@unittest.skipIf(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    "통합 테스트는 RUN_INTEGRATION_TESTS=true 환경 변수가 설정된 경우에만 실행됩니다."
)
class TestLlmGatewayIntegration(unittest.TestCase):
    """
    실제 LLM API를 호출하여 llm_gateway의 전체 흐름을 검증하는 통합 테스트.
    """
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스가 시작될 때 한 번만 .env 파일을 로드합니다."""
        load_dotenv()

    def test_call_llm_real_api_gpt(self):
        """[통합 테스트] GPT 모델(gpt-4o)을 실제로 호출하여 유효한 JSON을 받는지 테스트합니다."""
        
        # GIVEN: API 키 존재 여부 확인 및 실제 LLM에게 전달할 프롬프트
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY 환경 변수가 설정되지 않아 통합 테스트를 건너뜁니다.")

        # LLM이 JSON을 생성하도록 명확하게 지시하는 프롬프트
        prompt = """
        당신은 AI 어시스턴트입니다. 아래 지시에 따라 자기소개를 JSON 형식으로만 응답해주세요.
        다른 설명이나 대답은 일절 포함하지 마세요.
        - "name" 키에는 당신의 이름을 문자열로 넣어주세요.
        - "role" 키에는 당신의 역할을 문자열로 넣어주세요.
        """

        # WHEN: Mock 없이 실제 call_llm 함수를 실행
        result = call_llm(prompt, model="gpt-4o", temperature=0.1)

        # THEN: LLM의 응답이 비결정적이므로, 내용이 아닌 구조와 형식 위주로 검증
        print(f"\n[통합 테스트] LLM으로부터 받은 실제 응답:\n{result}")

        self.assertIsNotNone(result, "LLM 응답이 None입니다.")
        self.assertNotIn("error", result, f"LLM 호출 중 에러 발생: {result.get('error')}")
        self.assertIsInstance(result, dict, "LLM 응답이 딕셔너리 형태가 아닙니다.")
        
        # 프롬프트에서 요청한 키들이 실제로 포함되었는지 확인
        self.assertIn("name", result, "응답에 'name' 키가 없습니다.")
        self.assertIn("role", result, "응답에 'role' 키가 없습니다.")
        
        # 각 키의 값이 비어있지 않은 문자열인지 확인
        self.assertIsInstance(result["name"], str)
        self.assertGreater(len(result["name"]), 0, "'name' 값이 비어있습니다.")


if __name__ == '__main__':
    unittest.main()