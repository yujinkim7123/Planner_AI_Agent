import unittest
import copy
from validators.nlp.sna_validator import validate_sna_result

realistic_valid_sna_result = {
    "status": "SNA complete",
    "cluster_id": 2,
    "graph_data": {
        "nodes": [
            {"id": "가격", "community": 0},
            {"id": "품질", "community": 1},
            {"id": "배송", "community": 0}
        ],
        "links": [
            {"source": "가격", "target": "배송", "weight": 0.8},
            {"source": "품질", "target": "가격", "weight": 0.3}
        ]
    }
}
# -------------------------

class TestSnaValidator(unittest.TestCase):
    """SNA 결과 검증기에 대한 단위 테스트입니다."""

    def test_validation_succeeds_with_valid_data(self):
        """[성공 케이스] 유효한 SNA 결과가 검증을 통과하는지 테스트합니다."""
        print("\nRunning test: test_validation_succeeds_with_valid_data")
        result = validate_sna_result(realistic_valid_sna_result)
        self.assertIsNotNone(result)
        self.assertEqual(result['cluster_id'], 2)

    def test_validation_fails_with_missing_key(self):
        """[구조적 오류] 최상위 필수 키가 누락된 결과가 실패하는지 테스트합니다."""
        print("\nRunning test: test_validation_fails_with_missing_key")
        invalid_data = realistic_valid_sna_result.copy()
        del invalid_data["cluster_id"]
        result = validate_sna_result(invalid_data)
        self.assertIsNone(result)

    def test_validation_fails_with_invalid_nested_data(self):
        """[구조적 오류] graph_data에 잘못된 형태의 링크가 포함된 결과가 실패하는지 테스트합니다."""
        print("\nRunning test: test_validation_fails_with_invalid_nested_data")
        invalid_data = copy.deepcopy(realistic_valid_sna_result)
        del invalid_data["graph_data"]["links"][0]["source"]
        result = validate_sna_result(invalid_data)
        self.assertIsNone(result)

    def test_validation_fails_with_non_dict_input(self):
        """딕셔너리가 아닌 입력값이 None을 반환하는지 테스트합니다."""
        print("\nRunning test: test_validation_fails_with_non_dict_input")
        result = validate_sna_result("유효하지 않은 입력값")
        self.assertIsNone(result)

    # --- 아래 3개의 논리적 검증 테스트 케이스가 새로 추가되었습니다 ---

    def test_logical_fail_link_to_nonexistent_node(self):
        """[논리 오류] 존재하지 않는 노드를 가리키는 링크가 있을 때 실패하는지 테스트합니다."""
        print("\nRunning test: test_logical_fail_link_to_nonexistent_node")
        invalid_data = copy.deepcopy(realistic_valid_sna_result)
        invalid_data["graph_data"]["links"].append(
            {"source": "가격", "target": "존재하지않는노드", "weight": 0.5}
        )
        result = validate_sna_result(invalid_data)
        self.assertIsNone(result)

    def test_logical_fail_self_loop(self):
        """[논리 오류] 노드가 자기 자신을 가리키는 링크(셀프 루프)가 있을 때 실패하는지 테스트합니다."""
        print("\nRunning test: test_logical_fail_self_loop")
        invalid_data = copy.deepcopy(realistic_valid_sna_result)
        invalid_data["graph_data"]["links"].append(
            {"source": "가격", "target": "가격", "weight": 0.1}
        )
        result = validate_sna_result(invalid_data)
        self.assertIsNone(result)

    def test_logical_fail_non_positive_weight(self):
        """[논리 오류] 링크의 가중치(weight)가 0 이하일 때 실패하는지 테스트합니다."""
        print("\nRunning test: test_logical_fail_non_positive_weight")
        invalid_data = copy.deepcopy(realistic_valid_sna_result)
        invalid_data["graph_data"]["links"][0]["weight"] = 0.0 # 가중치를 0으로 변경
        result = validate_sna_result(invalid_data)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()