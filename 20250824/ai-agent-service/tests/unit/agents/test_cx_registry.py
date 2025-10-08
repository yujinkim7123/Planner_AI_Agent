# tests/agents/test_registry.py

import unittest
from unittest.mock import MagicMock

# 테스트 대상 모듈
from agents.experts.Analyst.registry import registry

class TestCXAnalystRegistry(unittest.TestCase):
    """CX Analyst 팀의 Registry 모듈의 논리적 정확성을 검증합니다."""

    def test_get_task_info_success(self):
        """(시나리오) 유효한 도구 이름으로 작업 정보를 성공적으로 조회하는지 테스트합니다."""
        task_info = registry.get_task_info("run_clustering")
        self.assertIsNotNone(task_info)
        self.assertIn("tool", task_info)
        self.assertIn("validator", task_info)
        self.assertEqual(task_info["payload_key"], "clustering")

    def test_get_task_info_failure(self):
        """(시나리오) 존재하지 않는 도구 이름으로 조회 시 ValueError를 발생하는지 테스트합니다."""
        with self.assertRaisesRegex(ValueError, "Registry에서 찾을 수 없습니다"):
            registry.get_task_info("run_non_existent_tool")

    def test_get_tools_description(self):
        """(시나리오) LLM을 위한 도구 설명 문자열을 올바르게 생성하는지 테스트합니다."""
        description = registry.get_tools_description()
        self.assertIsInstance(description, str)
        self.assertIn("run_clustering", description)
        self.assertIn("run_lda", description)
        self.assertIn("create_summary", description)

    def test_build_clustering_params(self):
        """_build_clustering_params가 파라미터를 정확히 구성하고, 데이터 부재 시 오류를 내는지 테스트합니다."""
        
        valid_state = {
            "retrieved_data_summary": {
                "top_documents_sample": [{"sentence_nouns": "디자인 성능"}, {"sentence_nouns": "가격 품질"}]
            }
        }
        params = registry._build_clustering_params(valid_state)
        self.assertEqual(len(params["documents"]), 2)
        self.assertEqual(params["documents"][0], "디자인 성능")
        

        invalid_state = {"retrieved_data_summary": {}}
        with self.assertRaisesRegex(ValueError, "문서 데이터가 없습니다"):
            registry._build_clustering_params(invalid_state)

    def test_build_lda_params(self):
        """_build_lda_params가 파라미터를 정확히 구성하고, 데이터 부재 시 오류를 내는지 테스트합니다."""
        
        valid_state = {
            "cx_insights": {"clustering": {"_temp_data": "mock_tfidf_matrix"}}
        }
        params = registry._build_lda_params(valid_state)
        self.assertEqual(params["temp_data"], "mock_tfidf_matrix")

    
        invalid_state = {"cx_insights": {}}
        with self.assertRaisesRegex(ValueError, "LDA를 위한 클러스터링 데이터가 없습니다."):
            registry._build_lda_params(invalid_state)

    def test_build_scores_params(self):
        """_build_scores_params가 파라미터를 정확히 구성하고, 데이터 부재 시 오류를 내는지 테스트합니다."""
  
        valid_state = {
            "retrieved_data_summary": {"top_documents_sample": ["doc1"]},
            "cx_insights": {"lda": {"topics_summary_list": ["topic1"]}}
        }
        params = registry._build_scores_params(valid_state)
        self.assertEqual(params["original_documents"], ["doc1"])
        self.assertEqual(params["lda_results"], {"topics_summary_list": ["topic1"]})

    
        invalid_state = {
            "retrieved_data_summary": {"top_documents_sample": ["doc1"]},
            "cx_insights": {}
        }
        with self.assertRaisesRegex(ValueError, "기회 점수 계산을 위한 LDA 데이터가 없습니다."):
            registry._build_scores_params(invalid_state)


    def test_build_summary_params(self):
        """_build_summary_params가 파라미터를 정확히 구성하고, 데이터 부재 시 오류를 내는지 테스트합니다."""
        
        valid_state = {
            "cx_insights": {"scores": ["score1", "score2"]}
        }
        params = registry._build_summary_params(valid_state)
        self.assertEqual(params["topics"], ["score1", "score2"])

        invalid_state = {"cx_insights": {}}
        with self.assertRaisesRegex(ValueError, "요약 생성을 위한 기회 점수\(topics\) 데이터가 없습니다\."):
            registry._build_summary_params(invalid_state)