import unittest
from validators.nlp.score_validator import validate_scores_result

realistic_valid_scores = [
    {"topic_id": "1-2", "action_keywords": ["소음"], "importance": 9.8, "satisfaction": 2.1, "opportunity_score": 17.7},
    {"topic_id": "0-1", "action_keywords": ["가격"], "importance": 8.5, "satisfaction": 4.5, "opportunity_score": 14.0}
]

invalid_score_out_of_range = [
    {"topic_id": "1-2", "action_keywords": ["소음"], "importance": 11.0, "satisfaction": 2.1, "opportunity_score": 18.9}
]
invalid_formula = [
    {"topic_id": "1-2", "action_keywords": ["소음"], "importance": 9.8, "satisfaction": 2.1, "opportunity_score": 99.9} # Wrong score
]
invalid_sort_order = [
    {"topic_id": "0-1", "action_keywords": ["가격"], "importance": 8.5, "satisfaction": 4.5, "opportunity_score": 14.0},
    {"topic_id": "1-2", "action_keywords": ["소음"], "importance": 9.8, "satisfaction": 2.1, "opportunity_score": 17.7} # Should be first
]
# --------------------------------

class TestScoreValidator(unittest.TestCase):
    """Unit tests for the enhanced opportunity score validator."""

    def test_succeeds_with_valid_and_logical_data(self):
        """[성공 케이스] 모든 논리 규칙을 통과하는 정상 데이터를 테스트합니다."""
        print("\nRunning test: test_succeeds_with_valid_and_logical_data")
        result = validate_scores_result(realistic_valid_scores)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_fails_with_score_out_of_range(self):
        """[논리 오류] 점수가 0-10 범위를 벗어날 때 실패하는지 테스트합니다."""
        print("\nRunning test: test_fails_with_score_out_of_range")
        result = validate_scores_result(invalid_score_out_of_range)
        self.assertIsNone(result)

    def test_fails_with_incorrect_formula(self):
        """[논리 오류] 기회 점수 계산 공식이 맞지 않을 때 실패하는지 테스트합니다."""
        print("\nRunning test: test_fails_with_incorrect_formula")
        result = validate_scores_result(invalid_formula)
        self.assertIsNone(result)

    def test_fails_with_incorrect_sort_order(self):
        """[논리 오류] 리스트가 내림차순 정렬되지 않았을 때 실패하는지 테스트합니다."""
        print("\nRunning test: test_fails_with_incorrect_sort_order")
        result = validate_scores_result(invalid_sort_order)
        self.assertIsNone(result)

    def test_fails_with_non_list_input(self):
        """[타입 오류] 입력값이 리스트가 아닐 때 실패하는지 테스트합니다."""
        print("\nRunning test: test_fails_with_non_list_input")
        result = validate_scores_result("not a list")
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()