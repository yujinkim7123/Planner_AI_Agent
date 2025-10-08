import unittest
from unittest.mock import patch, MagicMock

from tools.nlp.analysis_summary import run

class TestAnalysisSummary(unittest.TestCase):

    def setUp(self):
        """테스트에 사용될 샘플 토픽 데이터를 준비합니다."""
        self.sample_topics = [
            {"topic_id": "0-1", "action_keywords": ["가격", "오류"], "opportunity_score": 15.5},
            {"topic_id": "0-2", "action_keywords": ["디자인", "색상"], "opportunity_score": 12.3},
        ]

    @patch('tools.nlp.analysis_summary.call_llm')
    @patch('tools.nlp.analysis_summary.build_summary_prompt')
    def test_run_success(self, mock_build_prompt, mock_call_llm):
    
       
        mock_build_prompt.return_value = "This is a mock prompt for testing."
          
        mock_llm_response = {"summary": "This is a generated mock summary."}
        mock_call_llm.return_value = mock_llm_response

      
        result = run(self.sample_topics)

       
        mock_build_prompt.assert_called_once_with(self.sample_topics)
        
       
        mock_call_llm.assert_called_once_with(
            "This is a mock prompt for testing.", 
            model="gpt-4o", 
            temperature=0.1
        )
   
        self.assertEqual(result, mock_llm_response)

    @patch('tools.nlp.analysis_summary.call_llm')
    @patch('tools.nlp.analysis_summary.build_summary_prompt')
    def test_run_handles_llm_failure(self, mock_build_prompt, mock_call_llm):

      
        mock_build_prompt.return_value = "This is a mock prompt for testing."
        mock_call_llm.side_effect = Exception("LLM API Error")

        result = run(self.sample_topics)

        self.assertIsNone(result)

    @patch('tools.nlp.analysis_summary.call_llm')
    @patch('tools.nlp.analysis_summary.build_summary_prompt')
    def test_run_with_empty_topics(self, mock_build_prompt, mock_call_llm):
       
        empty_topics = []

        
        result = run(empty_topics)

       
        self.assertIsNone(result)
        
       
        mock_build_prompt.assert_not_called()
        mock_call_llm.assert_not_called()

if __name__ == '__main__':
    unittest.main()