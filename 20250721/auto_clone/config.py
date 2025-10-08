# 1_config.py
import os
import json

# 기본값 설정 (웹 인터페이스에서 값이 제공되지 않을 경우 대비)
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_KEYWORDS = {
    "portal": ["청소 후기", "청소 방법"],
    "blog": ["세탁기 사용법", "세탁기 후기"],
    "cafe": ["로봇청소기 중고"],
    "board": ["로봇청소기 추천"]
}
DEFAULT_TARGET_CAFE_URL = 'https://cafe.naver.com/robotclear'
DEFAULT_TARGET_BOARD_URL_FORMAT = "https://cafe.naver.com/f-e/cafes/13411273/menus/25?viewType=L&ta=SUBJECT&q={keyword}&page={page}&size=50"
DEFAULT_CRAWL_MAX_PAGES = 2
DEFAULT_CRAWL_START_DATE = '2022-01-01'
DEFAULT_CRAWL_END_DATE = '2025-07-07'

def load_config(config_data=None):
    """
    웹 인터페이스에서 전달된 설정을 로드하거나 기본값을 사용.
    config_data: dict, 웹에서 전달된 JSON 설정 (없으면 기본값 사용).
    """
    config = {}

    if config_data:
        # 웹에서 전달된 설정 사용
        config['QDRANT_HOST'] = config_data.get('QDRANT_HOST', DEFAULT_QDRANT_HOST)
        config['QDRANT_PORT'] = config_data.get('QDRANT_PORT', DEFAULT_QDRANT_PORT)
        config['KEYWORDS'] = config_data.get('KEYWORDS', DEFAULT_KEYWORDS)
        config['TARGET_CAFE_URL'] = config_data.get('TARGET_CAFE_URL', DEFAULT_TARGET_CAFE_URL)
        config['TARGET_BOARD_URL_FORMAT'] = config_data.get('TARGET_BOARD_URL_FORMAT', DEFAULT_TARGET_BOARD_URL_FORMAT)
        config['CRAWL_MAX_PAGES'] = config_data.get('CRAWL_MAX_PAGES', DEFAULT_CRAWL_MAX_PAGES)
        config['CRAWL_START_DATE'] = config_data.get('CRAWL_START_DATE', DEFAULT_CRAWL_START_DATE)
        config['CRAWL_END_DATE'] = config_data.get('CRAWL_END_DATE', DEFAULT_CRAWL_END_DATE)
    else:
        # 기본값 사용
        config['QDRANT_HOST'] = os.getenv('QDRANT_HOST', DEFAULT_QDRANT_HOST)
        config['QDRANT_PORT'] = int(os.getenv('QDRANT_PORT', DEFAULT_QDRANT_PORT))
        config['KEYWORDS'] = json.loads(os.getenv('KEYWORDS', json.dumps(DEFAULT_KEYWORDS)))
        config['TARGET_CAFE_URL'] = os.getenv('TARGET_CAFE_URL', DEFAULT_TARGET_CAFE_URL)
        config['TARGET_BOARD_URL_FORMAT'] = os.getenv('TARGET_BOARD_URL_FORMAT', DEFAULT_TARGET_BOARD_URL_FORMAT)
        config['CRAWL_MAX_PAGES'] = int(os.getenv('CRAWL_MAX_PAGES', DEFAULT_CRAWL_MAX_PAGES))
        config['CRAWL_START_DATE'] = os.getenv('CRAWL_START_DATE', DEFAULT_CRAWL_START_DATE)
        config['CRAWL_END_DATE'] = os.getenv('CRAWL_END_DATE', DEFAULT_CRAWL_END_DATE)

    # 네이버 인증 정보는 환경 변수에서만 로드 (보안상 하드코딩 제거)
    config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '')
    config['NAVER_ID'] = os.getenv('NAVER_ID', '')
    config['NAVER_PW'] = os.getenv('NAVER_PW', '')
    config['NAVER_API_CLIENT_ID'] = os.getenv('NAVER_API_CLIENT_ID', '')
    config['NAVER_API_CLIENT_SECRET'] = os.getenv('NAVER_API_CLIENT_SECRET', '')

    return config