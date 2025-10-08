# tools/nlp/score.py
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any


from models.sentiment_analyzer import analyze_sentiment

def _get_sentiment_score_from_result(result: Dict[str, Any]) -> float:
    """analyze_sentiment가 반환한 딕셔너리를 점수로 변환합니다."""
    if result.get('label') == 'positive':
        return float(result.get('score', 0.0))
    elif result.get('label') == 'negative':
        return -float(result.get('score', 0.0))
    return 0.0

def calculate_opportunity_scores(original_documents: List[Dict[str, Any]], lda_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    
    print(f"--- TOOL: 기회 점수 계산 실행 ---")

    lda_topics = lda_results.get("topics_summary_list", [])
    temp_data = lda_results.get("_temp_data", {})
    
    doc_indices_in_corpus = temp_data.get("document_indices_in_corpus", [])
    doc_primary_topic = temp_data.get("doc_primary_topic", [])

    if not all([lda_topics, doc_indices_in_corpus, doc_primary_topic]):
        raise ValueError("기회 점수 계산에 필요한 LDA 결과 데이터가 부족합니다.")

    # 1. 각 토픽에 어떤 문서들이 속하는지 매핑
    topic_to_docs_map = defaultdict(list)
    for i, topic_idx in enumerate(doc_primary_topic):
        original_doc_index = doc_indices_in_corpus[i]
        topic_id = lda_topics[topic_idx]["topic_id"]
        topic_to_docs_map[topic_id].append(original_documents[original_doc_index])

    # 2. 원시(raw) 중요도와 만족도 계산
    raw_scores = []
    for topic in lda_topics:
        topic_id = topic["topic_id"]
        docs_for_topic = topic_to_docs_map[topic_id]
        
        importance = len(docs_for_topic)

        # [핵심] 실제 감성 분석 모델을 호출하여 점수를 계산합니다.
        sentiment_results = [analyze_sentiment(doc.get('original_text', '')) for doc in docs_for_topic]
        satisfaction_scores = [_get_sentiment_score_from_result(res) for res in sentiment_results]
        satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 0.0
        
        raw_scores.append({
            "topic_id": topic_id,
            "action_keywords": topic["action_keywords"],
            "raw_importance": importance,
            "raw_satisfaction": satisfaction
        })

    # 3. 중요도와 만족도를 0~10점 척도로 정규화(Normalization)
    imp_values = [s["raw_importance"] for s in raw_scores]
    sat_values = [s["raw_satisfaction"] for s in raw_scores]
    imp_min, imp_max = min(imp_values) if imp_values else 0, max(imp_values) if imp_values else 0
    sat_min, sat_max = min(sat_values) if sat_values else 0, max(sat_values) if sat_values else 0

    def normalize(value, min_val, max_val):
        if max_val == min_val: return 5.0
        return (value - min_val) / (max_val - min_val) * 10.0

    # 4. 최종 기회 점수 계산
    final_scores = []
    for score in raw_scores:
        norm_imp = normalize(score["raw_importance"], imp_min, imp_max)
        norm_sat = normalize(score["raw_satisfaction"], sat_min, sat_max)
        opportunity_score = norm_imp + (10.0 - norm_sat)
        
        final_scores.append({
            "topic_id": score["topic_id"],
            "action_keywords": score["action_keywords"],
            "importance": round(norm_imp, 2),
            "satisfaction": round(norm_sat, 2),
            "opportunity_score": round(opportunity_score, 2)
        })

    # 5. 기회 점수가 높은 순으로 정렬하여 반환
    return sorted(final_scores, key=lambda x: x["opportunity_score"], reverse=True)