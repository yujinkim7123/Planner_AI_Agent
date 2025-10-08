# agents/utils.py
import torch
from sentence_transformers import SentenceTransformer, models
from qdrant_client import QdrantClient
from openai import OpenAI, AsyncOpenAI # AsyncOpenAI 임포트도 중요합니다!
import os 
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from qdrant_client.http.models import Filter, FieldCondition, MatchValue 
sentiment_analyzer = None 

# 모델과 클라이언트를 저장할 전역 변수
meaning_model = None
topic_model = None
qdrant_client = None
openai_client = None


def get_sentiment_analyzer():
    """
    [신규] 사전 학습된 한국어 감성 분류 모델을 로드합니다.
    Hugging Face의 pipeline을 사용하여 쉽게 구현합니다.
    """
    global sentiment_analyzer
    if sentiment_analyzer is None:
        print("🌀 Loading pre-trained sentiment analysis model...")
        
        # Define the local path to your model files
        local_model_path = "C:/Users/User/DIC_Project/persona_mcp_server/agents/models/bert-nsmc" 
        
        try:
            # Load the tokenizer from your local path
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)

            # 2. pipeline에 로컬 모델과 tokenizer 전달
            sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        except Exception as e:
            print(f"❌ Error loading sentiment analysis model from local path: {e}")
            sentiment_analyzer = None # Ensure it remains None if loading fails
    return sentiment_analyzer

def get_embedding_models():
    """
    의미 검색(e5-large) 모델과 주제 검색(ko-sbert) 모델을 모두 로드합니다.
    이미 로드되었다면 기존 객체를 반환합니다.
    """
    global meaning_model, topic_model
    if meaning_model is None or topic_model is None:
        print("🌀 Loading embedding models (meaning & topic)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Meaning Model 로드 (기존 코드와 동일)
        meaning_model = SentenceTransformer(
            modules=[models.Transformer("intfloat/e5-large"), models.Pooling(1024, pooling_mode_mean_tokens=True)],
            device=device
        )
        
        # 2. Topic Model 로드 (기존 코드와 동일)
        topic_model = SentenceTransformer("jhgan/ko-sbert-nli", device=device)
        
        print("✅ All embedding models loaded.")
        
    return meaning_model, topic_model

def get_qdrant_client():
    """Qdrant 클라이언트를 생성합니다."""
    global qdrant_client
    if qdrant_client is None:
        print("🌀 Initializing Qdrant client...")
        qdrant_client = QdrantClient(host="localhost", port=6333)
        print("✅ Qdrant client initialized.")
    return qdrant_client

def get_openai_client(async_client=False):
    """
    [개선됨] OpenAI 클라이언트를 반환합니다.
    최초 호출 시, 환경 변수에서 API 키를 읽어 객체를 생성합니다.
    """
    global openai_client
    if openai_client is None:
        print("🌀 Initializing OpenAI client...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("'.env' 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
        openai_client = OpenAI(api_key=api_key) # 항상 동기 클라이언트를 생성하고 전역 변수에 저장
    return openai_client # 저장된 동기 클라이언트를 반환


    
def parse_natural_date(text: str | None) -> tuple | None:
    """
    [최종 수정] '1년간', '3개월간' 등의 표현도 인식하도록 정규표현식을 수정합니다.
    """
    if not text:
        return None

    today = datetime.now()
    text = text.lower()
    
    match = re.search(r'(?:최근|지난)\s*(\d+)\s*(개월|달|년|주|일)간?', text)
    if match:
        num, unit = int(match.group(1)), match.group(2)
        end_date = today.date()
        if '년' in unit:
            start_date = (today - timedelta(days=num*365)).date()
        elif '주' in unit:
            start_date = (today - timedelta(days=num*7)).date()
        elif '일' in unit:
            start_date = (today - timedelta(days=num)).date()
        else: # 개월 또는 달
            start_date = (today - timedelta(days=num*30)).date()
        return start_date, end_date

    # 다른 패턴들은 그대로 유지
    if '올해' in text or '이번 년도' in text:
        return datetime(today.year, 1, 1).date(), today.date()
    if '작년' in text:
        last_year = today.year - 1
        return datetime(last_year, 1, 1).date(), datetime(last_year, 12, 31).date()

    if '이번 달' in text:
        return today.replace(day=1).date(), today.date()
    if '지난 달' in text:
        first_day_of_current_month = today.replace(day=1)
        last_day_of_last_month = first_day_of_current_month - timedelta(days=1)
        first_day_of_last_month = last_day_of_last_month.replace(day=1)
        return first_day_of_last_month.date(), last_day_of_last_month.date()

    if '어제' in text:
        yesterday = (today - timedelta(days=1)).date()
        return yesterday, yesterday
    if '오늘' in text:
        return today.date(), today.date()
        
    return None