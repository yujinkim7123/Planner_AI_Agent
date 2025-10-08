# 4_utils.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import numpy as np
import pandas as pd
from tqdm import tqdm
import uuid

# --- AI 모델 정의 (기존 코드와 동일) ---
class AttentiveHierarchicalClassifier(nn.Module):
    # ... (app.py에 있던 모델 클래스 정의 전체를 여기에 복사) ...
    def __init__(self, input_dim=769, hidden_dim=256, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.attn_doc = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.attn_global = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Linear(hidden_dim * 3, num_classes)

    def forward(self, x, doc_mask):
        h = self.encoder(x)
        doc_context = torch.zeros_like(h)
        unique_doc_ids = torch.unique(doc_mask)
        for doc_id_val in unique_doc_ids:
            doc_indices_tuple = (doc_mask == doc_id_val).nonzero(as_tuple=True)
            if doc_indices_tuple[0].numel() > 0:
                doc_indices = doc_indices_tuple[0]
                doc_h = h[doc_indices]
                weights = torch.softmax(self.attn_doc(doc_h).squeeze(-1), dim=0)
                weighted = torch.sum(doc_h * weights.unsqueeze(-1), dim=0, keepdim=True)
                doc_context[doc_indices] = weighted.expand(doc_indices.size(0), -1)
        global_weights = torch.softmax(self.attn_global(h).squeeze(-1), dim=0)
        global_context = torch.sum(h * global_weights.unsqueeze(-1), dim=0, keepdim=True)
        expanded_global = global_context.expand(h.size(0), -1)
        concat = torch.cat([h, doc_context, expanded_global], dim=1)
        logits = self.classifier(concat)
        return logits

class InferenceDataset(Dataset):
    # ... (app.py에 있던 데이터셋 클래스 정의 전체를 여기에 복사) ...
    def __init__(self, df, embedding_model):
        self.all_inputs = []
        self.doc_indexes = []
        unique_doc_ids = df['doc_id'].unique()
        doc_id_map = {doc_id: idx for idx, doc_id in enumerate(unique_doc_ids)}
        
        for doc_id, group in tqdm(df.groupby("doc_id"), desc="데이터셋 구성"):
            sentences = group['sentence'].tolist()
            embeddings = embedding_model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
            sentence_ids = group['sentence_idx'].values.astype(np.float32)
            sentence_ids -= sentence_ids.min()
            max_val = sentence_ids.max()
            if len(sentence_ids) > 1 and max_val > 0: sentence_ids = sentence_ids / max_val
            else: sentence_ids = np.zeros_like(sentence_ids)
            positions = sentence_ids[:, np.newaxis]
            inputs = np.concatenate([embeddings, positions], axis=1)
            doc_index = doc_id_map[doc_id]
            self.all_inputs.extend(inputs)
            self.doc_indexes.extend([doc_index] * len(group))
        self.all_inputs = torch.tensor(np.array(self.all_inputs), dtype=torch.float32)
        self.doc_indexes = torch.tensor(np.array(self.doc_indexes), dtype=torch.long)

    def __len__(self): return len(self.all_inputs)
    def __getitem__(self, idx): return self.all_inputs[idx], self.doc_indexes[idx]

class NLPModels:
    """모든 NLP 모델을 로드하고 관리하는 싱글톤 클래스."""
    _instance = None

    def __new__(cls, model_path='best_model.pt'):
        if cls._instance is None:
            cls._instance = super(NLPModels, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path='best_model.pt'):
        if self._initialized:
            return
        
        print("NLP 모델을 로딩합니다...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.meaning_model = SentenceTransformer("intfloat/e5-large", device=self.device)
        self.topic_model = SentenceTransformer("jhgan/ko-sbert-nli", device=self.device)
        self.okt = Okt()
        self.classifier = self._load_classifier(model_path)
        self._initialized = True
        print("NLP 모델 로딩 완료!")

    def _load_classifier(self, model_path):
        model = AttentiveHierarchicalClassifier()
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # torch.compile 등으로 인해 _orig_mod. 접두사가 붙는 경우 제거
            from collections import OrderedDict
            new_state_dict = OrderedDict((k[10:] if k.startswith('_orig_mod.') else k, v) for k, v in state_dict.items())
            model.load_state_dict(new_state_dict)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"분류기 모델 로딩 실패: {e}. AI 필터링을 건너뜁니다.")
            return None

    def create_inference_dataset(self, df):
        return InferenceDataset(df, self.topic_model)

    def predict(self, dataset):
        if not self.classifier or len(dataset) == 0:
            return pd.Series([1] * len(dataset)) # 모델 없으면 모두 통과
            
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="AI 모델 예측"):
                inputs, doc_masks = batch
                inputs, doc_masks = inputs.to(self.device), doc_masks.to(self.device)
                logits = self.classifier(inputs, doc_masks)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
        return all_preds
        
    def extract_nouns(self, text):
        nouns = self.okt.nouns(text)
        return ' '.join([n for n in nouns if len(n) > 1])
        
    def get_rule_based_keywords(self):
        # app.py의 get_keyword_lists 함수와 동일한 내용
        return {"ad": ["상품", "옵션", "ml", "세트", "최저가격"], "ad_descriptive": ["장점", "단점", "기능", "효과적"], "url": ["http", "com", "smartstore"], "fiction": [], "religious": []}


class QdrantManager:
    """Qdrant DB와의 모든 상호작용을 관리하는 클래스."""
    def __init__(self, host, port, nlp_models):
        self.client = QdrantClient(host=host, port=port)
        self.nlp = nlp_models

    def upload_data(self, df, collection_name):
        if df.empty:
            print("업로드할 데이터가 없습니다.")
            return
            
        if self.client.collection_exists(collection_name=collection_name):
            self.client.delete_collection(collection_name=collection_name)
            print(f"기존 컬렉션 '{collection_name}' 삭제 완료.")
            
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "meaning": VectorParams(size=self.nlp.meaning_model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
                "topic": VectorParams(size=self.nlp.topic_model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
            }
        )
        print(f"컬렉션 '{collection_name}' 생성 완료.")

        points = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"'{collection_name}'에 데이터 업로드"):
            vectors = {
                "meaning": self.nlp.meaning_model.encode("query: " + row["sentence"]),
                "topic": self.nlp.topic_model.encode(row["sentence_nouns"])
            }
            payload = {
                "sentence": row["sentence"], "sentence_nouns": row["sentence_nouns"],
                "doc_id": row["doc_id"], "sentence_idx": row["sentence_idx"],
                "date": str(row["date"]), "url": row["url"]
            }
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vectors, payload=payload))

        self.client.upsert(collection_name=collection_name, points=points, wait=True)
        print(f"🎉 총 {len(points)}개의 데이터가 '{collection_name}' 컬렉션에 업로드되었습니다.")