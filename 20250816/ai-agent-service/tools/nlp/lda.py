# tools/nlp/lda.py
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, PCA
from scipy.sparse import csr_matrix

# --- 내부 헬퍼(보조) 함수 ---
def _get_top_keywords(feature_names, topic_components, n_top_words):
    """LDA 토픽 모델에서 각 토픽별 상위 N개 키워드를 추출합니다."""
    top_keywords = []
    for topic_idx, topic in enumerate(topic_components):
        top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
        keywords_for_topic = [feature_names[i] for i in top_words_indices]
        top_keywords.append(keywords_for_topic)
    return top_keywords

# --- 공개 기술 (Public Tool) ---
def run_lda(temp_data: dict, cluster_id: int, num_topics: int = 3) -> dict:
    """
    특정 클러스터에 대해 LDA 토픽 모델링을 수행하고, 상세 결과를 반환합니다.
    (기존 run_lda_tool의 로직과 동일)
    """
    print(f"--- TOOL: LDA 토픽 모델링 실행 (Cluster ID: {cluster_id}) ---")
    
    cluster_labels = temp_data.get("cluster_labels", [])
    tfidf_matrix = np.array(temp_data.get("tfidf_matrix", []))
    feature_names = temp_data.get("feature_names", [])

    docs_indices_in_corpus = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
    if not docs_indices_in_corpus or len(docs_indices_in_corpus) < num_topics:
         raise ValueError(f"클러스터 {cluster_id}의 문서 수가 토픽 수보다 적습니다.")

    doc_term_matrix = csr_matrix(tfidf_matrix[docs_indices_in_corpus])
    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    doc_topic_dist = lda.fit_transform(doc_term_matrix)
    
    top_keywords_per_topic = _get_top_keywords(feature_names, lda.components_, 7)

    topic_positions_2d = []
    if lda.components_.shape[0] >= 2:
        pca = PCA(n_components=2, random_state=42)
        topic_positions_2d = pca.fit_transform(lda.components_).tolist()
    
    topics_list = []
    for i, keywords in enumerate(top_keywords_per_topic):
        topic_info = {
            "topic_id": f"{cluster_id}-{i}",
            "action_keywords": keywords,
        }
        if i < len(topic_positions_2d):
            topic_info["position_2d"] = {"x": topic_positions_2d[i][0], "y": topic_positions_2d[i][1]}
        
        topics_list.append(topic_info)
        
    return {
        "status": "LDA complete",
        "cluster_id": cluster_id,
        "num_topics": num_topics,
        "topics_summary_list": topics_list,
        "_temp_data": {
            "lda_topics": topics_list,
            "document_indices_in_corpus": docs_indices_in_corpus,
            "doc_topic_dist": doc_topic_dist.tolist(),
            "doc_primary_topic": np.argmax(doc_topic_dist, axis=1).tolist()
        }
    }