from typing import List, Dict, Any

def build_summary_prompt(topics: List[Dict[str, Any]]) -> str:
    
    # 기회 점수가 높은 상위 5개 토픽만 요약에 사용
    top_topics = topics[:5]
    
    # LLM에게 전달할 토픽 데이터 문자열 생성
    topics_str = "\n".join([
        f"- 토픽: '{', '.join(t['action_keywords'])}' (중요도: {t['importance']:.1f}, 만족도: {t['satisfaction']:.1f}, 기회점수: {t['opportunity_score']:.1f})"
        for t in top_topics
    ])

    prompt = f"""당신은 CX 데이터 분석 전문가입니다. 아래는 고객 VOC 데이터의 핵심 토픽과 기회 점수 분석 결과입니다.
    이 데이터를 바탕으로 현재 고객 경험의 어떤 부분이 가장 중요한 기회 영역인지 한두 문장으로 간결하게 요약해주세요.

    [분석 결과]
    {topics_str}

    [요약]
    """
    return prompt