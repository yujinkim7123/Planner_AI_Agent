# main.py
import asyncio
from crawlers import PortalCrawler, BlogCrawler, CafeCrawler, BoardCrawler
from processors import PortalBlogProcessor, CafeProcessor
from utils import NLPModels, QdrantManager
from config import load_config

async def process_source(source_type, config, log_callback=None):
    """지정된 소스에 대한 크롤링 및 처리 파이프라인을 실행."""
    logs = []
    def log(message):
        logs.append(message)
        if log_callback:
            log_callback(message)

    log(f"\n{'='*20} [{source_type.upper()}] 프로세스 시작 {'='*20}")

    # 1. 크롤러와 프로세서 선택
    if source_type == "portal":
        crawler = PortalCrawler(keywords=config["KEYWORDS"]["portal"])
        processor = PortalBlogProcessor(nlp_models, qdrant_manager)
        collection_name = "naver_portal_data"
    elif source_type == "blog":
        crawler = BlogCrawler(
            keywords=config["KEYWORDS"]["blog"],
            client_id=config["NAVER_API_CLIENT_ID"],
            client_secret=config["NAVER_API_CLIENT_SECRET"]
        )
        processor = PortalBlogProcessor(nlp_models, qdrant_manager)
        collection_name = "naver_blog_data"
    elif source_type == "cafe":
        crawler = CafeCrawler(
            keywords=config["KEYWORDS"]["cafe"],
            cafe_url=config["TARGET_CAFE_URL"],
            naver_id=config["NAVER_ID"],
            naver_pw=config["NAVER_PW"],
            start_date=config["CRAWL_START_DATE"],
            end_date=config["CRAWL_END_DATE"],
            max_pages=config["CRAWL_MAX_PAGES"]
        )
        processor = CafeProcessor(nlp_models, qdrant_manager)
        collection_name = "naver_cafe_data"
    elif source_type == "board":
        crawler = BoardCrawler(
            keywords=config["KEYWORDS"]["board"],
            board_url_format=config["TARGET_BOARD_URL_FORMAT"],
            naver_id=config["NAVER_ID"],
            naver_pw=config["NAVER_PW"]
        )
        processor = CafeProcessor(nlp_models, qdrant_manager)
        collection_name = "naver_board_data"
    else:
        log(f"'{source_type}'은(는) 유효한 소스 타입이 아닙니다.")
        return logs

    # 2. 데이터 크롤링
    log(f"[{source_type}] 데이터 수집을 시작합니다...")
    raw_df = await crawler.crawl()
    
    if raw_df.empty:
        log(f"[{source_type}] 수집된 데이터가 없습니다. 프로세스를 종료합니다.")
        return logs

    log(f"[{source_type}] 데이터 수집 완료. 총 {len(raw_df)}개의 문서 수집.")
    
    # 3. 데이터 처리 및 DB 업로드
    processor.run_pipeline(raw_df, collection_name)
    
    log(f"✅ [{'='*20} [{source_type.upper()}] 프로세스 완료 {'='*20}]")
    return logs

# 공용 유틸리티 (FastAPI에서 초기화)
nlp_models = NLPModels()
qdrant_manager = None  # FastAPI에서 동적으로 초기화