# tools/nlp/sna.py
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import community as community_louvain

def run_sna(temp_data: dict, cluster_id: int) -> dict:
    """
    클러스터링 결과 중 특정 클러스터에 대해 의미 연결망 분석(SNA)을 수행합니다.
    (기존 run_sna_tool의 로직과 동일)
    """
    print(f"--- TOOL: SNA 실행 (Cluster ID: {cluster_id}) ---")
    
    cluster_labels = temp_data.get("cluster_labels", [])
    tfidf_matrix = np.array(temp_data.get("tfidf_matrix", []))
    feature_names = temp_data.get("feature_names", [])

    docs_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
    if not docs_indices:
        raise ValueError(f"ID가 {cluster_id}인 클러스터에 문서가 없습니다.")

    cluster_matrix = csr_matrix(tfidf_matrix[docs_indices])
    co_occurrence_matrix = (cluster_matrix.T * cluster_matrix)
    co_occurrence_matrix.setdiag(0)
    
    G = nx.from_scipy_sparse_array(co_occurrence_matrix)
    
    mapping = {i: name for i, name in enumerate(feature_names)}
    G = nx.relabel_nodes(G, mapping)
    
    partitions = community_louvain.best_partition(G)
    
    graph_data = nx.node_link_data(G)
    for node in graph_data['nodes']:
        node['community'] = partitions.get(node['id'], 0)

    return {
        "status": "SNA complete",
        "cluster_id": cluster_id,
        "graph_data": graph_data
    }