import unittest
from validators.nlp.cluster_validator import validate_clustering_result
import copy

# 개선된 Validator를 통과할 수 있도록 description 필드 추가 등 상세화된 테스트 데이터
valid_clustering_output = {
    "status": "Clustering complete", "num_clusters": 2,
    "cluster_labels": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    "cluster_summaries": {
        "0": {"keywords": ["가격", "가성비", "할인"], "num_docs": 4, "description": "가격 관련 클러스터"},
        "1": {"keywords": ["세척력", "소음", "건조", "성능"], "num_docs": 6, "description": "성능 관련 클러스터"}
    },
    "visual_data": {
        "reduced_features_2d": [[0.1, 0.2]] * 10, 
        "cluster_labels": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    },
    "temp_data": {
        "tfidf_matrix": [[0.5] * 10] * 10, 
        "feature_names": ["가격", "성능"], 
        "cluster_labels": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    }
}


class TestLogicalClusterValidator(unittest.TestCase):

    def test_validation_succeeds_with_consistent_data(self):
        """[성공] 논리적으로 완벽한 데이터가 검증을 성공적으로 통과하는지 테스트합니다."""
        print("\nRunning test: test_validation_succeeds_with_consistent_data")
        result = validate_clustering_result(valid_clustering_output)
        self.assertIsNotNone(result)
        self.assertEqual(result['num_clusters'], 2)

    def test_validation_fails_with_inconsistent_summary_count(self):
        """[기존 논리 오류] num_clusters와 summaries 개수가 다를 때 검증이 실패하는지 테스트합니다."""
        print("\nRunning test: test_validation_fails_with_inconsistent_summary_count")
        invalid_data = copy.deepcopy(valid_clustering_output)
        invalid_data["cluster_summaries"] = {"0": valid_clustering_output["cluster_summaries"]["0"]} # 요약을 1개로 줄임
        result = validate_clustering_result(invalid_data)
        self.assertIsNone(result)

    def test_validation_fails_with_inconsistent_labels(self):
        """[기존 논리 오류] 여러 위치의 cluster_labels 데이터가 서로 다를 때 검증이 실패하는지 테스트합니다."""
        print("\nRunning test: test_validation_fails_with_inconsistent_labels")
        invalid_data = copy.deepcopy(valid_clustering_output)
        invalid_data["temp_data"]["cluster_labels"] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] # 레이블 불일치 생성
        result = validate_clustering_result(invalid_data)
        self.assertIsNone(result)
        
    def test_validation_fails_with_mismatched_doc_count_sum(self):
        """[신규 논리 오류] summaries의 num_docs 총합이 실제 문서 수와 다를 때 검증이 실패하는지 테스트합니다."""
        print("\nRunning test: test_validation_fails_with_mismatched_doc_count_sum")
        invalid_data = copy.deepcopy(valid_clustering_output)
        # 문서 총합이 10이 아닌 9가 되도록 수정
        invalid_data["cluster_summaries"]["0"]["num_docs"] = 3 
        result = validate_clustering_result(invalid_data)
        self.assertIsNone(result)

    def test_validation_fails_with_invalid_label_range(self):
        """[신규 논리 오류] cluster_labels에 허용 범위를 벗어난 값이 있을 때 검증이 실패하는지 테스트합니다."""
        print("\nRunning test: test_validation_fails_with_invalid_label_range")
        invalid_data = copy.deepcopy(valid_clustering_output)
        # num_clusters가 2이므로 0, 1만 유효. 3은 유효하지 않은 레이블.
        invalid_labels_with_outlier = [3, 0, 1, 1, 0, 1, 0, 1, 0, 1] 
        invalid_data["cluster_labels"] = invalid_labels_with_outlier
        invalid_data["visual_data"]["cluster_labels"] = invalid_labels_with_outlier
        invalid_data["temp_data"]["cluster_labels"] = invalid_labels_with_outlier
        result = validate_clustering_result(invalid_data)
        self.assertIsNone(result)

    def test_validation_fails_with_missing_key(self):
        """[구조적 오류] 필수 키가 누락되었을 때 검증이 실패하는지 테스트합니다."""
        print("\nRunning test: test_validation_fails_with_missing_key")
        structurally_invalid = valid_clustering_output.copy()
        del structurally_invalid["status"]
        result = validate_clustering_result(structurally_invalid)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()