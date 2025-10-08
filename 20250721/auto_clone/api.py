# api.py
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from config import load_config
from main import process_source, nlp_models
from utils import QdrantManager
import asyncio

app = FastAPI()

class CrawlConfig(BaseModel):
    QDRANT_HOST: str
    QDRANT_PORT: int
    KEYWORDS: dict
    TARGET_CAFE_URL: str
    TARGET_BOARD_URL_FORMAT: str
    CRAWL_MAX_PAGES: int
    CRAWL_START_DATE: str
    CRAWL_END_DATE: str
    NAVER_ID: str
    NAVER_PW: str
    NAVER_API_CLIENT_ID: str
    NAVER_API_CLIENT_SECRET: str
    sources: list[str]

# 정적 HTML 파일 제공
try:
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
except FileNotFoundError:
    html_content = "<h1>Error: index.html not found</h1>"
except UnicodeDecodeError as e:
    html_content = f"<h1>Error: Failed to decode index.html - {str(e)}</h1>"

@app.get("/")
async def get():
    return HTMLResponse(html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async def log_callback(message):
        await websocket.send_text(message)
    
    try:
        # WebSocket으로부터 설정 수신
        config_data = await websocket.receive_json()
        config = load_config(config_data)
        
        # QdrantManager 초기화
        global qdrant_manager
        qdrant_manager = QdrantManager(host=config["QDRANT_HOST"], port=config["QDRANT_PORT"], nlp_models=nlp_models)
        
        # 크롤링 실행
        for source in config_data["sources"]:
            logs = await process_source(source, config, log_callback)
            for log in logs:
                await websocket.send_text(log)
        
        await websocket.send_text("🎉 모든 크롤링 작업 완료!")
    except Exception as e:
        await websocket.send_text(f"오류 발생: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)