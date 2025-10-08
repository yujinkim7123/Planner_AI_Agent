import unittest
import copy
from validators.nlp.lda_validator import validate_lda_result


def get_valid_lda_data():
    """모든 검증을 통과하는 유효한 기본 데이터셋을 반환합니다."""
    return {
        "status": "LDA complete",
        "cluster_id": 10,
        "num_topics": 2,
        "topics_summary_list": [
            {"topic_id": "10-0", "action_keywords": ["키워드A", "키워드B"]},
            {"topic_id": "10-1", "action_keywords": ["키워드C", "키워드D"]},
        ],
        "visual_data": {
            "topic_positions_2d": [[0.1, 0.2], [-0.1, -0.2]],
            "doc_topic_dist": [
                [0.8, 0.2],  # 합계 1.0
                [0.1, 0.9],  # 합계 1.0
                [0.5, 0.5],  # 합계 1.0
            ],
        },
        "temp_data": {
            "document_indices_in_corpus": [5, 15, 25], # 문서 3개
            "doc_primary_topic": [0, 1, 1], # 0과 1만 포함
        },
    }

# --- Unittest 테스트 케이스 클래스 ---

class TestLdaValidator(unittest.TestCase):

    def test_successful_validation(self):
        """[성공] 모든 데이터가 유효하고 논리적으로 일관될 때 검증이 성공하는지 테스트합니다."""
        valid_data = get_valid_lda_data()
        result = validate_lda_result(valid_data)
        self.assertIsNotNone(result)
        self.assertEqual(result, valid_data)

    def test_fails_on_mismatched_topic_count(self):
        """[실패] num_topics와 topics_summary_list의 길이가 다를 때 실패하는지 테스트합니다."""
        invalid_data = get_valid_lda_data()
        invalid_data["num_topics"] = 3 # 길이 불일치 유발
        result = validate_lda_result(invalid_data)
        self.assertIsNone(result)

    def test_fails_on_duplicate_topic_id(self):
        """[실패] topic_id가 중복될 때 실패하는지 테스트합니다."""
        invalid_data = get_valid_lda_data()
        invalid_data["topics_summary_list"][1]["topic_id"] = "10-0" # ID 중복 유발
        result = validate_lda_result(invalid_data)
        self.assertIsNone(result)

    def test_fails_on_invalid_topic_id_format(self):
        """[실패] topic_id 형식이 cluster_id와 맞지 않을 때 실패하는지 테스트합니다."""
        invalid_data = get_valid_lda_data()
        invalid_data["topics_summary_list"][0]["topic_id"] = "99-0" # 형식 오류 유발
        result = validate_lda_result(invalid_data)
        self.assertIsNone(result)

    def test_fails_on_positions_2d_length_mismatch(self):
        """[실패] topic_positions_2d의 길이가 num_topics와 다를 때 실패하는지 테스트합니다."""
        invalid_data = get_valid_lda_data()
        invalid_data["visual_data"]["topic_positions_2d"] = [[0.1, 0.2]] # 길이 불일치 유발
        result = validate_lda_result(invalid_data)
        self.assertIsNone(result)

    def test_fails_on_doc_topic_dist_row_mismatch(self):
        """[실패] doc_topic_dist의 행 수가 실제 문서 수와 다를 때 실패하는지 테스트합니다."""
        invalid_data = get_valid_lda_data()
        invalid_data["visual_data"]["doc_topic_dist"] = [[0.8, 0.2]] # 행 수 불일치 유발
        result = validate_lda_result(invalid_data)
        self.assertIsNone(result)

    def test_fails_on_doc_topic_dist_col_mismatch(self):
        """[실패] doc_topic_dist의 열 수가 num_topics와 다를 때 실패하는지 테스트합니다."""
        invalid_data = get_valid_lda_data()
        invalid_data["visual_data"]["doc_topic_dist"][0] = [0.7, 0.2, 0.1] # 열 수 불일치 유발
        result = validate_lda_result(invalid_data)
        self.assertIsNone(result)

    def test_fails_on_doc_topic_dist_sum_error(self):
        """[실패] doc_topic_dist의 확률 합계가 1이 아닐 때 실패하는지 테스트합니다."""
        invalid_data = get_valid_lda_data()
        invalid_data["visual_data"]["doc_topic_dist"][1] = [0.1, 0.8] # 합계 0.9, 오류 유발
        result = validate_lda_result(invalid_data)
        self.assertIsNone(result)

    def test_fails_on_primary_topic_length_mismatch(self):
        """[실패] doc_primary_topic의 길이가 실제 문서 수와 다를 때 실패하는지 테스트합니다."""
        invalid_data = get_valid_lda_data()
        invalid_data["temp_data"]["doc_primary_topic"] = [0, 1] # 길이 불일치 유발
        result = validate_lda_result(invalid_data)
        self.assertIsNone(result)

    def test_fails_on_invalid_primary_topic_value(self):
        """[실패] doc_primary_topic에 유효하지 않은 토픽 번호가 있을 때 실패하는지 테스트합니다."""
        invalid_data = get_valid_lda_data()
        invalid_data["temp_data"]["doc_primary_topic"] = [0, 1, 2] # 유효 범위(0,1) 벗어남
        result = validate_lda_result(invalid_data)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()