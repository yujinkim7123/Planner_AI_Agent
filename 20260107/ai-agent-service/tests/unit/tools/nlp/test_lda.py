import unittest
import numpy as np
# 테스트할 도구(run_lda)와 그 결과를 검증할 함수(validate_lda_result)를 모두 가져옵니다.
from tools.nlp.lda import run_lda
from validators.nlp.lda_validator import validate_lda_result

class TestLdaTool(unittest.TestCase):
    """LDA 토픽 모델링 도구에 대한 단위 테스트입니다."""

    def setUp(self):
        """
        테스트를 위해 run_clustering의 결과물인 temp_data를 시뮬레이션합니다.
        총 10개 문서, 2개 클러스터(0번: 가격, 1번: 성능) 상황을 가정합니다.
        """
        self.feature_names = ["가격", "할인", "가성비", "성능", "세척력", "소음"]
        self.cluster_labels = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0] # 0번 5개, 1번 5개
        
        # 10개 문서 x 6개 특징에 대한 가상 TF-IDF 행렬
        # 0-4번 문서는 가격 관련 단어에, 5-9번 문서는 성능 관련 단어에 높은 가중치를 줍니다.
        tfidf_matrix = np.zeros((10, 6))
        tfidf_matrix[0:5, 0:3] = np.random.rand(5, 3) # 가격 클러스터
        tfidf_matrix[5:10, 3:6] = np.random.rand(5, 3) # 성능 클러스터
        
        self.mock_temp_data = {
            "feature_names": self.feature_names,
            "cluster_labels": self.cluster_labels,
            "tfidf_matrix": tfidf_matrix.tolist()
        }

    def test_successful_run_passes_validation(self):
        """[성공 케이스] 정상적인 입력으로 LDA 실행 시, 결과가 validator를 통과하는지 테스트합니다."""
        print("\nRunning test: test_successful_run_passes_validation")
        cluster_to_test = 0
        num_topics_to_find = 2
        
        # 1. 도구를 실행하여 실제 결과물을 받습니다.
        result_from_tool = run_lda(
            self.mock_temp_data, 
            cluster_id=cluster_to_test, 
            num_topics=num_topics_to_find
        )
        
        # 2. 결과물을 엄격한 validator 함수에 통과시킵니다.
        validated_result = validate_lda_result(result_from_tool)
        
        # 3. validator가 None을 반환하지 않으면 모든 구조적/논리적 검증을 통과한 것입니다.
        self.assertIsNotNone(validated_result)
        self.assertEqual(validated_result['cluster_id'], cluster_to_test)
        self.assertEqual(validated_result['num_topics'], num_topics_to_find)
        
    def test_insufficient_documents_for_topics_raises_error(self):
        """[실패 케이스] 클러스터의 문서 수가 토픽 수보다 적을 때 ValueError가 발생하는지 테스트합니다."""
        print("\nRunning test: test_insufficient_documents_for_topics_raises_error")
        # 0번 클러스터에는 5개의 문서가 있는데, 6개의 토픽을 찾으려고 시도
        with self.assertRaises(ValueError):
            run_lda(self.mock_temp_data, cluster_id=0, num_topics=6)

    def test_no_documents_for_cluster_raises_error(self):
        """[실패 케이스] 분석할 클러스터 ID에 해당하는 문서가 없을 때 ValueError가 발생하는지 테스트합니다."""
        print("\nRunning test: test_no_documents_for_cluster_raises_error")
        # 존재하지 않는 99번 클러스터를 분석하려고 시도
        with self.assertRaises(ValueError):
            run_lda(self.mock_temp_data, cluster_id=99, num_topics=2)

if __name__ == '__main__':
    unittest.main()