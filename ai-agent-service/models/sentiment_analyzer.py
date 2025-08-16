# models/sentiment_analyzer.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

#전역변수로 초기에 모델 로드
ANALYZER = None
try:
    print("감성 분석 모델을 로드합니다... (matthew-c/korean-sentiment-analysis-base)")
    MODEL_NAME = "matthew-c/korean-sentiment-analysis-base"
    
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
    """
    로드된 감성 분석기(pipeline)를 반환
    """
    return ANALYZER

def analyze_sentiment(text: str) -> dict:
    """
    주어진 텍스트에 대해 감성 분석을 수행하고, 결과를 딕셔너리로 반환
    """
    analyzer = get_sentiment_analyzer()
    if not analyzer:
        # 모델 로드 실패 시, 기본값을 반환
        return {"label": "neutral", "score": 0.0}
        
    try:
        # 모델 입력 길이 제한(512)에 맞춰 텍스트를 자름
        truncated_text = text[:512]
        result = analyzer(truncated_text)[0]
        return result 
    except Exception as e:
        print(f"감성 점수 계산 중 오류 발생: {e}")
        return {"label": "error", "score": 0.0}