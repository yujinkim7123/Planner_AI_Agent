import unittest
import os
from dotenv import load_dotenv

from tools.nlp.score import calculate_opportunity_scores
from validators.nlp.score_validator import validate_scores_result

@unittest.skipIf(
    os.getenv("RUN_INTEGRATION_TESTS") != "true",
    "통합 테스트는 RUN_INTEGRATION_TESTS=true 환경 변수가 설정된 경우에만 실행됩니다."
)
class TestScoreIntegration(unittest.TestCase):
    """
    실제 감성 분석 모델을 호출하여 기회 점수 계산 전체 흐름을 검증하는 통합 테스트입니다.
    """

    @classmethod
    def setUpClass(cls):
        """테스트 클래스가 시작될 때 한 번만 .env 파일을 로드합니다."""
        load_dotenv()

    def setUp(self):
        """테스트에 사용될 원본 문서와 LDA 결과 데이터를 준비합니다."""
        self.sample_documents = [
            # --- 부정적인 문서 (토픽 0) ---
            {"doc_id": "doc_01", "original_text": "가격이 너무 비싸고 전혀 만족스럽지 않아요."},
            {"doc_id": "doc_02", "original_text": "이 가격에 이런 품질이라니 실망입니다."},
            {"doc_id": "doc_03", "original_text": "가격 대비 성능이 매우 나쁩니다."},
            # --- 긍정적인 문서 (토픽 1) ---
            {"doc_id": "doc_04", "original_text": "디자인이 정말 예쁘고 마음에 쏙 들어요."},
            {"doc_id": "doc_05", "original_text": "배송도 빠르고 디자인도 세련됐습니다. 최고예요!"},
            {"doc_id": "doc_06", "original_text": "화면과 똑같이 디자인이 아름답습니다. 만족해요."},
        ]

        
        self.sample_lda_results = {
            "topics_summary_list": [
                {"topic_id": "cluster0-0", "action_keywords": ["가격", "품질", "성능"]},
                {"topic_id": "cluster0-1", "action_keywords": ["디자인", "배송", "색상"]},
            ],
            "_temp_data": {
                "document_indices_in_corpus": [0, 1, 2, 3, 4, 5],
                "doc_primary_topic": [0, 0, 0, 1, 1, 1],
            }
        }

    def test_score_calculation_with_real_model(self):
        """[통합 테스트] 실제 감성분석 모델을 연동하여 기회 점수 계산을 테스트합니다."""
        
        
        result_scores = calculate_opportunity_scores(self.sample_documents, self.sample_lda_results)

        print(f"\n[통합 테스트] 실제 모델로 계산된 최종 기회 점수:\n{result_scores}")

        
        validated_result = validate_scores_result(result_scores)
        self.assertIsNotNone(validated_result, "기회 점수 결과가 validator 검증을 통과하지 못했습니다.")
        
       
        price_topic = next(item for item in validated_result if "가격" in item["action_keywords"])
        design_topic = next(item for item in validated_result if "디자인" in item["action_keywords"])

      
        self.assertGreater(design_topic['satisfaction'], price_topic['satisfaction'])
       
        self.assertGreater(price_topic['opportunity_score'], design_topic['opportunity_score'])
      
        self.assertEqual(validated_result[0]['topic_id'], price_topic['topic_id'])


if __name__ == '__main__':
    unittest.main()