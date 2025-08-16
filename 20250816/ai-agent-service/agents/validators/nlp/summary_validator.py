from typing import Any, Optional

def validate_summary_result(result: Any) -> Optional[str]:

    if not isinstance(result, str):
        print("Validation failed: 요약 결과가 문자열(str)이 아닙니다.")
        return None
        
    if not result.strip():
        print("Validation failed: 요약 결과가 비어있습니다.")
        return None
        
    if len(result) < 10:
        print(f"Validation failed: 요약 결과가 너무 짧습니다 (길이: {len(result)}).")
        return None

    print("Validation passed: 요약문이 유효합니다.")
    return result