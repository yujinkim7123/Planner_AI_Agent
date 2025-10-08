import unittest
import numpy as np

# 실제 프로젝트 구조에 맞게 경로를 조정해주세요.
from tools.nlp.cluster import run_clustering
from tools.nlp.sna import run_sna
from validators.nlp.sna_validator import validate_sna_result

class TestSnaIntegration(unittest.TestCase):

    def setUp(self):
        """테스트에 사용될 실제와 유사한 문서 데이터를 준비합니다."""
        self.sample_documents = [
            "가격은 좋은데 품질이 아쉬워요.",
            "디자인은 예쁜데 배송이 너무 느려요.",
            "품질도 좋고 가격도 합리적이네요.",
            "배송 상태는 좋았지만 디자인이 별로예요.",
            "이 가격에 이 정도 품질이라니 만족합니다.",
            "색상은 예쁜데 배송이 파손되어서 왔어요."
        ]
        self.num_clusters = 2

    def test_sna_with_real_clustering_output(self):
        print("\nRunning test: test_sna_with_real_clustering_output")

        clustering_result = run_clustering(self.sample_documents, num_clusters=self.num_clusters)
        self.assertIsNotNone(clustering_result, "선행 작업인 클러스터링 실행에 실패했습니다.")
        
        temp_data_from_clustering = clustering_result.get("temp_data")
        self.assertIsNotNone(temp_data_from_clustering, "클러스터링 결과에 temp_data가 없습니다.")
        
        cluster_to_test = 0

        sna_result = run_sna(temp_data_from_clustering, cluster_id=cluster_to_test)
        
        validated_result = validate_sna_result(sna_result)
        
        print(f"\n[통합 테스트] SNA 결과 (클러스터 {cluster_to_test}):\n{validated_result}")
        
        self.assertIsNotNone(validated_result, "SNA 결과가 논리적 검증을 통과하지 못했습니다.")
        self.assertEqual(validated_result['cluster_id'], cluster_to_test)
        self.assertIn('nodes', validated_result['graph_data'])
        self.assertIn('links', validated_result['graph_data'])
        self.assertGreater(len(validated_result['graph_data']['nodes']), 0, "SNA 결과에 노드가 없습니다.")


if __name__ == '__main__':
    unittest.main()