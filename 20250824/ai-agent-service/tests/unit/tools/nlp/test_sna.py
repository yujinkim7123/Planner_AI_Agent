import unittest
import numpy as np
from tools.nlp.sna import run_sna
from validators.nlp.sna_validator import validate_sna_result

class TestSnaTool(unittest.TestCase):
    """SNA(의미 연결망 분석) 도구에 대한 단위 테스트입니다."""

    def setUp(self):
        self.feature_names = ["가격", "품질", "배송"]
        self.cluster_labels = [0, 1, 0, 1] # 0번 2개, 1번 2개
        
        # 가상 TF-IDF 행렬 (정수형으로 변경하여 명확한 가중치 보장)
        # 0, 2번 문서: 가격, 배송 키워드
        # 1, 3번 문서: 품질, 가격 키워드
        tfidf_matrix = np.array([
            [5, 0, 3], # doc 0
            [4, 6, 0], # doc 1
            [6, 0, 4], # doc 2
            [5, 7, 0]  # doc 3
        ])
        
        self.mock_temp_data = {
            "feature_names": self.feature_names,
            "cluster_labels": self.cluster_labels,
            "tfidf_matrix": tfidf_matrix.tolist()
        }

    def test_successful_run_passes_validation(self):
        """[성공 케이스] 정상 입력으로 SNA 실행 시, 결과가 validator를 통과하는지 테스트합니다."""
        print("\nRunning test: test_successful_run_passes_validation")
        cluster_to_test = 0
        
        # 1. 도구를 실행하여 실제 결과물을 받습니다.
        result_from_tool = run_sna(self.mock_temp_data, cluster_id=cluster_to_test)
        
        # 2. 결과물을 논리적 검증이 포함된 validator 함수에 통과시킵니다.
        validated_result = validate_sna_result(result_from_tool)
        
        # 3. validator가 None을 반환하지 않으면 모든 검증을 통과한 것입니다.
        self.assertIsNotNone(validated_result, "run_sna의 결과가 논리적 검증을 통과하지 못했습니다.")
        self.assertEqual(validated_result['cluster_id'], cluster_to_test)
        # 그래프 데이터의 기본 구조 확인
        self.assertIn('nodes', validated_result['graph_data'])
        self.assertIn('links', validated_result['graph_data'])
        
    def test_no_documents_for_cluster_raises_error(self):
        """[실패 케이스] 분석할 클러스터 ID에 해당하는 문서가 없을 때 ValueError가 발생하는지 테스트합니다."""
        print("\nRunning test: test_no_documents_for_cluster_raises_error")
        # 존재하지 않는 99번 클러스터를 분석하려고 시도
        with self.assertRaises(ValueError):
            run_sna(self.mock_temp_data, cluster_id=99)

if __name__ == '__main__':
    unittest.main()