# tests/agents/test_preconditions.py

import unittest

from agents.experts.Analyst.validators.preconditions import check

class TestCXAnalystPreconditions(unittest.TestCase):
    """CX Analyst 팀의 사전 조건 검증(preconditions) 모듈의 논리를 테스트합니다."""

    def test_check_all_conditions_met(self):
        """(시나리오 1) 모든 데이터가 준비된 상태에서는 모든 검사가 통과하는지 테스트합니다."""
       
        full_state = {
            "retrieved_data_summary": {
                "top_documents_sample": ["doc1", "doc2"]
            },
            "cx_insights": {
                "clustering": {"status": "complete"},
                "lda": {"status": "complete"},
                "scores": [{"score": 10}]
            }
        }
        
       
        is_runnable, reason = check(full_state, "run_clustering")
        self.assertTrue(is_runnable)

        is_runnable, reason = check(full_state, "run_lda")
        self.assertTrue(is_runnable)

        is_runnable, reason = check(full_state, "run_sna")
        self.assertTrue(is_runnable)

        is_runnable, reason = check(full_state, "calculate_scores")
        self.assertTrue(is_runnable)

    def test_check_fails_for_clustering(self):
        """(시나리오 2) 'retrieved_data_summary'가 없을 때 clustering 검사가 실패하는지 테스트합니다."""
       
        initial_state = {}
        
      
        is_runnable, reason = check(initial_state, "run_clustering")
        
       
        self.assertFalse(is_runnable)
        self.assertIn("'retrieved_data_summary' 데이터가 먼저 제공되어야 합니다.", reason)

    def test_check_fails_for_lda_and_sna(self):
        """(시나리오 3) 'clustering' 결과가 없을 때 lda와 sna 검사가 실패하는지 테스트합니다."""
       
        state_after_retrieval = {
            "retrieved_data_summary": {
                "top_documents_sample": ["doc1", "doc2"]
            },
            "cx_insights": {} 
        }
        
    
        is_runnable_lda, reason_lda = check(state_after_retrieval, "run_lda")
        self.assertFalse(is_runnable_lda)
        self.assertIn("'클러스터링'이 먼저 수행되어야 합니다.", reason_lda)

     
        is_runnable_sna, reason_sna = check(state_after_retrieval, "run_sna")
        self.assertFalse(is_runnable_sna)
        self.assertIn("'클러스터링'이 먼저 수행되어야 합니다.", reason_sna)

    def test_check_fails_for_scores(self):
        """(시나리오 4) 'lda' 결과가 없을 때 scores 계산 검사가 실패하는지 테스트합니다."""
    
        state_after_clustering = {
            "retrieved_data_summary": {"top_documents_sample": ["doc1"]},
            "cx_insights": {
                "clustering": {"status": "complete"}
               
            }
        }
        
        is_runnable, reason = check(state_after_clustering, "calculate_scores")
        

        self.assertFalse(is_runnable)
        self.assertIn("'LDA 토픽 모델링'이 먼저 수행되어야 합니다.", reason)