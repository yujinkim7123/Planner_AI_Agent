import unittest
from unittest.mock import patch
# 실제 프로젝트 구조에 맞게 경로를 조정해주세요.
from tools.nlp.score import calculate_opportunity_scores
from validators.nlp.score_validator import validate_scores_result

class TestOpportunityScoreCalculator(unittest.TestCase):

    def setUp(self):
        """각 테스트 메서드 실행 전에 필요한 기본 데이터셋을 준비합니다."""
        self.sample_documents = [
            {"doc_id": "doc_01", "original_text": "가격이 너무 비싸요."},
            {"doc_id": "doc_02", "original_text": "디자인은 정말 예뻐요."},
            {"doc_id": "doc_03", "original_text": "가성비가 좋지 않아요."},
            {"doc_id": "doc_04", "original_text": "디자인이 세련됐습니다."},
            {"doc_id": "doc_05", "original_text": "색감이 마음에 들어요."},
            {"doc_id": "doc_06", "original_text": "가격만 빼면 완벽해요."},
        ]
        self.sample_lda_results = {
            "topics_summary_list": [
                {"topic_id": "0-0", "action_keywords": ["가격", "가성비"]},
                {"topic_id": "0-1", "action_keywords": ["디자인", "색감", "세련"]},
            ],
            "_temp_data": {
                "document_indices_in_corpus": [0, 1, 2, 3, 4, 5],
                "doc_primary_topic": [0, 1, 0, 1, 1, 0],
            }
        }

    @patch('tools.nlp.score.analyze_sentiment')
    def test_calculate_opportunity_scores_success(self, mock_analyze_sentiment):
        """
        [성공] 기회 점수 계산이 성공적으로 실행되고, validator 검증을 통과하는지 테스트합니다.
        """
        
        def mock_sentiment_analyzer(text):
            if "비싸요" in text or "좋지 않아요" in text:
                return {'label': 'negative', 'score': 0.9}
            elif "예뻐요" in text or "세련됐습니다" in text or "마음에 들어요" in text or "완벽해요" in text:
                return {'label': 'positive', 'score': 0.8}
            return {'label': 'neutral', 'score': 0.0}
        
        mock_analyze_sentiment.side_effect = mock_sentiment_analyzer

       
        result = calculate_opportunity_scores(self.sample_documents, self.sample_lda_results)

       
        validated_result = validate_scores_result(result)
        self.assertIsNotNone(validated_result)
        self.assertEqual(len(validated_result), 2)

     
        price_topic = next(item for item in validated_result if "가격" in item["action_keywords"])
        design_topic = next(item for item in validated_result if "디자인" in item["action_keywords"])
        
        self.assertGreater(price_topic["opportunity_score"], design_topic["opportunity_score"])
        
        self.assertEqual(validated_result[0], price_topic)

    def test_calculate_opportunity_scores_value_error(self):
        """
        [실패] 기회 점수 계산에 필요한 LDA 데이터가 부족할 때 ValueError가 발생하는지 테스트합니다.
        """
      
        invalid_lda_results = {
            "topics_summary_list": [],
            "_temp_data": {}
        }

       
        with self.assertRaises(ValueError) as cm:
            calculate_opportunity_scores(self.sample_documents, invalid_lda_results)
        
        self.assertIn("기회 점수 계산에 필요한 LDA 결과 데이터가 부족합니다.", str(cm.exception))


if __name__ == '__main__':
    unittest.main()