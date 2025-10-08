# tools/nlp/cluster.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


def run_clustering_tool(documents: list[str], num_clusters: int = 5) -> dict:
    """
    주어진 문서들에 대해 Ward 클러스터링을 수행하고, 분석에 필요한 모든 중간 데이터를 반환합니다.
    """
    print(f"--- TOOL: {num_clusters}개 클러스터링 실행 ---")
    if not documents or len(documents) < num_clusters:
        raise ValueError("클러스터링을 위한 문서 수가 부족하거나 클러스터 개수보다 적습니다.")

    noun_stopwords = ['것', '수', '일', '문제', '경우', '생각', '사용', '기능', '제품', '정보'] # 예시

    vectorizer = TfidfVectorizer(max_features=2000, min_df=2, stop_words=noun_stopwords)
    X = vectorizer.fit_transform(documents)

    if X.shape[0] < 2:
        raise ValueError("유효한 문서가 부족하여 클러스터링을 진행할 수 없습니다.")

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, n_init='auto', batch_size=256)
    kmeans.fit(X)
    
    cluster_labels = kmeans.labels_.tolist()
    feature_names = vectorizer.get_feature_names_out()
    
    cluster_summaries = {}
    for i in range(num_clusters):
        cluster_docs_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_docs_indices) > 0:
            cluster_tfidf_sum = X[cluster_docs_indices].sum(axis=0)
            top_feature_indices = np.asarray(cluster_tfidf_sum).flatten().argsort()[-10:][::-1]
            top_keywords = [feature_names[idx] for idx in top_feature_indices]
            cluster_summaries[str(i)] = {
                "keywords": top_keywords,
                "num_docs": int(len(cluster_docs_indices)),
                "description": f"{i}번 그룹 ({len(cluster_docs_indices)}개 문서)은 '{', '.join(top_keywords[:3])}'와 관련 있습니다."
            }
        else:
            cluster_summaries[str(i)] = {"keywords": [], "num_docs": 0, "description": "문서 없음"}


    pca = PCA(n_components=2, random_state=42)
    reduced_features_2d = pca.fit_transform(X.toarray()).tolist()

    return {
        "status": "Clustering complete",
        "num_clusters": num_clusters,
        "cluster_labels": cluster_labels,
        "cluster_summaries": cluster_summaries,
        "visual_data": { "reduced_features_2d": reduced_features_2d, "cluster_labels": cluster_labels },
        "_temp_data": {
            "tfidf_matrix": X.toarray().tolist(),
            "feature_names": feature_names.tolist(),
            "cluster_labels": cluster_labels
        }
    }
