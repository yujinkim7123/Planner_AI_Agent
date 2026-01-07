import unittest
from unittest.mock import patch, MagicMock
import importlib

import models.sentiment_analyzer as sentiment_analyzer

class TestSentimentAnalyzer(unittest.TestCase):

    def test_successful_model_load_and_analysis(self):
        """
        [성공] 모델 로딩이 성공했을 때, analyze_sentiment가 정상 작동하는지 테스트합니다.
        """
        # GIVEN: transformers 라이브러리의 함수들을 Mock 객체로 대체하여 성공 시나리오를 시뮬레이션
        mock_pipeline = MagicMock(return_value=[{'label': 'LABEL_1', 'score': 0.99}])
        
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model, \
             patch('transformers.pipeline', return_value=mock_pipeline) as mock_pipeline_factory:

            # WHEN: Mock이 적용된 상태에서 모듈을 다시 로드하여 전역 변수를 초기화
            importlib.reload(sentiment_analyzer)
            
            # and: 감성 분석 함수를 호출
            result = sentiment_analyzer.analyze_sentiment("이 서비스는 정말 최고예요!")

            # THEN: Mock 파이프라인이 올바른 텍스트로 호출되었는지 확인
            mock_pipeline.assert_called_once_with("이 서비스는 정말 최고예요!"[:512])
            
            # and: 반환된 결과가 Mock 파이프라인의 결과와 일치하는지 확인
            self.assertEqual(result, {'label': 'LABEL_1', 'score': 0.99})

    def test_model_load_failure(self):
        """
        [실패] 모델 로딩 중 예외가 발생했을 때, analyze_sentiment가 기본값을 반환하는지 테스트합니다.
        """
        # GIVEN: 모델 로딩 함수가 예외를 발생시키도록 설정
        with patch('transformers.AutoModelForSequenceClassification.from_pretrained', 
                   side_effect=Exception("Model not found")):
            
            # WHEN: Mock이 적용된 상태에서 모듈을 다시 로드하면, 내부적으로 except 블록이 실행됨
            importlib.reload(sentiment_analyzer)
            
            # and: 감성 분석 함수를 호출
            result = sentiment_analyzer.analyze_sentiment("아무 텍스트")

            # THEN: 모델 로드가 실패했으므로, 기본 중립값을 반환해야 함
            self.assertEqual(result, {"label": "neutral", "score": 0.0})

    def test_analysis_time_error(self):
        """
        [실패] 모델은 로드됐지만, 특정 텍스트 분석 중 오류 발생 시 기본값을 반환하는지 테스트합니다.
        """
        # GIVEN: 파이프라인 객체는 존재하지만, 호출 시 예외를 발생시키도록 설정
        mock_pipeline = MagicMock(side_effect=Exception("Analysis failed"))

        with patch('transformers.AutoTokenizer.from_pretrained'), \
             patch('transformers.AutoModelForSequenceClassification.from_pretrained'), \
             patch('transformers.pipeline', return_value=mock_pipeline):

            # WHEN: 모듈을 다시 로드하고
            importlib.reload(sentiment_analyzer)
            
            # and: 감성 분석 함수를 호출
            result = sentiment_analyzer.analyze_sentiment("오류를 유발하는 텍스트")

            # THEN: 분석 중 오류가 발생했으므로, 기본 에러값을 반환해야 함
            self.assertEqual(result, {"label": "error", "score": 0.0})

    def test_text_truncation(self):
        """
        [성공] 입력 텍스트가 512자 이상일 때, 잘라서 처리하는지 테스트합니다.
        """
        # GIVEN: 성공적으로 로드된 Mock 파이프라인과 512자가 넘는 텍스트
        mock_pipeline = MagicMock(return_value=[{'label': 'LABEL_0', 'score': 0.95}])
        long_text = "a" * 600
        
        with patch('transformers.pipeline', return_value=mock_pipeline):
            importlib.reload(sentiment_analyzer)
            
            # WHEN: 긴 텍스트로 감성 분석 함수를 호출
            sentiment_analyzer.analyze_sentiment(long_text)

            # THEN: Mock 파이프라인이 512자로 잘린 텍스트로 호출되었는지 확인
            mock_pipeline.assert_called_once_with(long_text[:512])


if __name__ == '__main__':
    unittest.main()