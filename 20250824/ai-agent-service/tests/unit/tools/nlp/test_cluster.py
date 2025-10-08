import unittest
# 테스트할 도구(run_clustering)와, 그 결과를 검증할 함수(validate_clustering_result)를 모두 가져옵니다.
from tools.nlp.cluster import run_clustering
from validators.nlp.cluster_validator import validate_clustering_result

class TestClusterToolWithValidator(unittest.TestCase):

    def setUp(self):
        """테스트를 위한 풍부하고 현실적인 샘플 문서들을 준비합니다."""
        self.documents =  [
            # 주제 1: 가격 및 가성비
            "가격 곳 저렴 구매 카드 할인 이벤트",
            "성능 대비 가성비 제품 생각 만족",
            "특별 할인 기간 구매 가격 마음",
            "가치 후회",
            "이벤트 쿠폰 포인트 사용 합리 가격",

            # 주제 2: 성능 및 기능
            "세척력 기대 이상 강력 빨래",
            "밤 사용 소음",
            "스마트 기능 앱 예약 세탁",
            "건조 기능 걱정 기능",
            "이전 모델 탈수 성능",
            "아이 옷 살균 코스 안심 세척력",
            "밤 세탁기 소음 때문 걱정 필요",

            # 주제 3: 배송 및 설치
            "주문 다음 날 배송",
            "설치 기사 약속 시간 설명",
            "배송 희망일 도착 계획 차질",
            "기사 설치 위치 수평",
            "배송 외부 긁힘"
        ]
        
    def test_successful_run_passes_logical_validation(self):
        print("\nRunning test: test_successful_run_passes_logical_validation")
        num_clusters = 3
        
        # 1. 도구를 실행하여 실제 결과물을 받습니다.
        result_from_tool = run_clustering(self.documents, num_clusters=num_clusters)
        #print(result_from_tool)
        # 2. 결과물을 엄격한 validator 함수에 통과시킵니다.
        validated_result = validate_clustering_result(result_from_tool)
        
        # 3. validator가 None을 반환하지 않았다면, 모든 구조적/논리적 검증을 통과한 것입니다.
        self.assertIsNotNone(validated_result, "run_clustering의 결과가 논리적 검증을 통과하지 못했습니다.")
        self.assertEqual(validated_result['num_clusters'], num_clusters)
        
    def test_insufficient_documents_raises_error(self):
        """[실패 케이스] 문서 수가 클러스터 수보다 적을 때 ValueError가 발생하는지 테스트합니다."""
        print("\nRunning test: test_insufficient_documents_raises_error")
        few_documents = ["문서 하나", "문서 둘"]
        num_clusters = 3
        
        with self.assertRaises(ValueError):
            run_clustering(few_documents, num_clusters=num_clusters)

if __name__ == '__main__':
    unittest.main()