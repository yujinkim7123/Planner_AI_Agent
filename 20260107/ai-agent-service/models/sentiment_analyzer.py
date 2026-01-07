# models/sentiment_analyzer.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

#전역변수로 초기에 모델 로드
ANALYZER = None
try:
    print("감성 분석 모델을 로드합니다... (sangrimlee/bert-base-multilingual-cased-nsmc)")
    MODEL_NAME = "sangrimlee/bert-base-multilingual-cased-nsmc"
    
    #tokenizer_config.json에 auto_map 설정이 없어 직접 클래스를 지정
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    ANALYZER = pipeline(
        "sentiment-analysis",
        model=MODEL,
        tokenizer=TOKENIZER
    )
    print("✅ 감성 분석 모델 로드 완료.")
except Exception as e:
    print(f"❌ 감성 분석 모델 로드 실패: {e}")
    print("감성 분석 기능이 비활성화됩니다.")

def get_sentiment_analyzer():
    return ANALYZER

def analyze_sentiment(text: str) -> dict:

    analyzer = get_sentiment_analyzer()
    if not analyzer:
        return {"label": "neutral", "score": 0.0}
        
    try:
        truncated_text = text[:512]
        result = analyzer(truncated_text)[0]
        return result 
    except Exception as e:
        print(f"감성 점수 계산 중 오류 발생: {e}")
        return {"label": "error", "score": 0.0}