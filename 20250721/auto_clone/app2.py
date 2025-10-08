# app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import uuid
import time
from datetime import datetime, timedelta
import openai
from kss import split_sentences
from sentence_transformers import SentenceTransformer
import logging

# ✅ [1] 라이브러리 추가: 형태소 분석기 Okt import
from konlpy.tag import Okt

# Selenium 관련 라이브러리
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
import random
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc

# --- Page Configuration & Styling ---
st.set_page_config(page_title="잠재고객 발굴 및 웹 데이터 처리 시스템", page_icon="🚀", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .st-emotion-cache-16txtl3 { padding-top: 2rem; }
    .st-emotion-cache-1avcm0n {
        background-color: #ffffff; border-radius: 10px; padding: 25px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #0062ff; color: white; border-radius: 5px; border: none; padding: 10px 24px;
        text-align: center; font-size: 16px; margin: 4px 2px; cursor: pointer; transition-duration: 0.4s;
    }
    .stButton>button:hover { background-color: #004ecb; }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions & Core Logic ---

# ✅ [2] NLP 모델 로딩 통합 함수 (효율성 증대)
@st.cache_resource
def load_nlp_models():
    """NLP에 필요한 모든 모델(임베딩, 형태소 분석기)을 한번에 로드합니다."""
    with st.spinner("NLP 모델(SentenceTransformer, Okt)을 로딩하고 있습니다..."):
        models = {
            "meaning": SentenceTransformer("intfloat/e5-large"),
            "topic": SentenceTransformer("jhgan/ko-sbert-nli"),
            "okt": Okt()
        }
    st.success("✅ NLP 모델 로딩 완료!")
    return models

# ✅ [3] 명사 추출 헬퍼 함수
def extract_nouns(text, okt_tagger):
    """Okt 형태소 분석기를 사용해 텍스트에서 명사를 추출합니다."""
    nouns = okt_tagger.nouns(text)
    nouns = [n for n in nouns if len(n) > 1] # 한 글자 명사 제외
    return ' '.join(nouns)

# ⭐️ --- 노트북 로직 반영: 신규 추가된 전처리 함수들 --- ⭐️
def filter_comments(comments_series):
    """
    .ipynb 파일의 댓글 처리 로직을 적용합니다.
    10글자 미만의 댓글을 제거하고, 남은 댓글이 없으면 '댓글 없음'으로 처리합니다.
    """
    def process_single_comment_string(x):
        if not isinstance(x, str):
            return '댓글 없음'
        
        # .ipynb에서는 '"'로 분리했지만, 크롤러가 ','로 수집하므로 이를 기준으로 분리
        comment_list = [comment.strip() for comment in x.split(',') if comment.strip()]
        
        # 10글자 이상인 댓글만 필터링
        filtered_list = [c for c in comment_list if len(c) >= 10]
        
        if not filtered_list:
            return '댓글 없음'
        else:
            return ' '.join(filtered_list)

    return comments_series.apply(process_single_comment_string)

def remove_repetitive_phrases(review_series):
    """
    .ipynb 파일에서 정의된 반복 문구를 제거합니다.
    """
    phrases_to_remove = [
        "경상도 3pl 전문 라온아토입니다", "(원스톱 사업자등록) 월 18000원 비상주 서비스 바로 가입! 바로 임대차계약서 발급까지!",
        "출석합니다", "출첵합니다.", "가입인사", "잘부탁드립니다.", "감사합니다.", "부탁드립니다", "연락부탁드립니다.",
        "긴글+ 사진폭탄 주의", "질문 있습니당.", "★", "댓글 없음",
        "< 아래 양식을 지켜주셔야 글 삭제와 회원강등이 안되니 꼭 지켜서 작성해주세요 >",
        "양식을 무시할 경우 운영규칙상 삭제 및 활동정지 될수 있으니 참고부탁드립니다.",
        "인스타 양도글은 인터넷 진흥원에서 안된다고 경고를 해서 카페에서 금지하고 있습니다",
        "카페활동 없이 양도글만 남길 경우 사이트의 자세한 설명없이 개인연락 유도글은 제재하고 있습니다",
        "안녕하세요", "안녕하세요!", "안녕하세요.", "쪽지드렸습니다", "챗드릴게요", "쪽지주세요", "쪽지드릴게요", "쪽지 드렸어요",
        "댓글이나 쪽지로 주세요", " 카페 활동 공지 사항 필독 <-- 가입인사 글쓰기 후 확인하시고 활동하셔야 합니다",
        "◆◆◆◆◆◆    진단평가와 관련된 내용만 올려주세요 그 외의 글은 삭제 및 활동제재합니다 ◆◆◆◆◆",
        "구합니다.", "등업조건:가입인사,출석 게시판 각 1개씩 글을 쓰고 각 게시판에 10씩 댓글을 달면 자동등업",
        "판매합니다", "수고하세요.", "쪽지 부탁드려요.", "소통해요.", "소통하고싶어요.", "해드리겠습니다.", "진행해드리겠습니다.",
        "말씀드리겠습니다.", "모집중에있습니다.", "양도합니다.", "전매합니다.", "확인합니다", "댓글없음.", "나눔후기\\)", "\\[체험후기\\]"
    ]
    
    # Series의 str.replace를 반복적으로 사용하여 문구 제거
    cleaned_series = review_series.copy()
    for phrase in phrases_to_remove:
        cleaned_series = cleaned_series.str.replace(phrase, "", regex=False)
    return cleaned_series

def apply_final_text_clean(review_series):
    """
    .ipynb 파일의 마지막 텍스트 정제 로직을 적용합니다. (한글, 영어, 공백 제외 모두 제거)
    """
    series = review_series.str.replace("[0-9]+갤","개월", regex = True)
    series = series.str.replace("[^가-힣a-zA-Z ]","", regex=True)
    series = series.str.replace("\\s+", " ", regex=True)
    series = series.str.strip()
    return series
# ⭐️ --- 노트북 로직 반영 끝 --- ⭐️


# AI 모델 클래스 정의 (기존과 동일)
class AttentiveHierarchicalClassifier(nn.Module):
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

# AI 모델 로딩 및 추론을 위한 함수 (기존과 동일)
@st.cache_resource
def load_inference_model(model_path, device):
    """best_model.pt를 로드하고, torch.compile로 인해 생긴 접두사를 처리합니다."""
    model = AttentiveHierarchicalClassifier(input_dim=769, hidden_dim=256, num_classes=2)
    try:
        state_dict = torch.load(model_path, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                name = k[10:]
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        st.success(f"`{model_path}` 모델 로딩 및 키 이름 정규화 성공!")
        return model
    except FileNotFoundError:
        st.error(f"모델 파일을 찾을 수 없습니다: '{model_path}'. app.py와 같은 폴더에 위치시켜주세요.")
        return None
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        return None

class InferenceDataset(Dataset):
    def __init__(self, df, embedding_model):
        self.all_inputs = []
        self.doc_indexes = []
        unique_doc_ids = df['doc_id'].unique()
        doc_id_map = {doc_id: idx for idx, doc_id in enumerate(unique_doc_ids)}

        with st.spinner("모델 입력을 위한 데이터셋 구성 중..."):
            for doc_id, group in tqdm(df.groupby("doc_id"), desc="문서 그룹 처리"):
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

def predict_with_model(model, dataset, device, batch_size=1024):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    model.eval()
    with torch.no_grad(), st.spinner("AI 모델이 문장별 광고 여부를 예측하고 있습니다..."):
        for batch in tqdm(dataloader, desc="모델 추론"):
            inputs, doc_masks = batch
            inputs, doc_masks = inputs.to(device), doc_masks.to(device)

            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                logits = model(inputs, doc_masks)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds

# 크롤링, 텍스트 정제, Rule 기반 필터링 함수들...
def execute_cafe_crawl_procedure(main_keywords, clean_key, max_items_per_keyword):
    # ... (기존 크롤링 코드 전체, 수정 없음) ...
    # 크롤러는 board_titles를 수집하지 않으므로, 해당 필터링은 현재 비활성화됩니다.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-notifications")
    options.add_argument('--disable-popup-blocking')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    st.info("Selenium WebDriver를 설정합니다. 처음 실행 시 시간이 걸릴 수 있습니다.")
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    except Exception as e:
        st.error(f"WebDriver 설정 중 오류가 발생했습니다: {e}")
        st.error("이 시스템은 로컬 환경에서 Chrome 브라우저가 설치되어야 정상 작동합니다.")
        return pd.DataFrame()
    eliminate_cafe = [ "https://cafe.naver.com/root74", 'https://cafe.naver.com/glycosarang', 'https://cafe.naver.com/appleiphone', 'https://cafe.naver.com/clubng', 'https://cafe.naver.com/outletshop', 'https://cafe.naver.com/gjtnwls123', 'https://cafe.naver.com/movie567', 'https://cafe.naver.com/komandos', 'https://cafe.naver.com/comicity', 'https://cafe.naver.com/bk1009', 'https://cafe.naver.com/bestofmobile', 'https://cafe.naver.com/illnesses', 'https://cafe.naver.com/simsll', 'https://cafe.naver.com/nexonsfw', 'https://cafe.naver.com/ipod5', 'https://cafe.naver.com/goldenpeach', 'https://cafe.naver.com/itop5', 'https://cafe.naver.com/peopledisc', 'https://cafe.naver.com/soho', 'https://cafe.naver.com/tictoc', 'https://cafe.naver.com/s7942', 'https://cafe.naver.com/godofhighschooljjang', 'https://cafe.naver.com/joonggonara', 'https://cafe.naver.com/hongdaeholic', 'https://cafe.naver.com/poohstory' ]
    eliminate_ads_keyword = ('-광고 -홍보 -할인 -핫딜 -특가 -이벤트 -수수료 -제휴 -공구 -딜 -가이드 -문의 -상담 -카톡 -카카오톡 -전화 -대표번호 -입점 -업체 -판매처 -도매 -총판 -직거래 -사은품 -한정수량 -재고확보 -최저가 -최대혜택 -무료체험 -체험단 -무료나눔 -구매링크 -정품보장 -루이비통 -구찌 -명품 -쇼핑몰 -도매가 -배송대행 -명품직구 -스타일링 -해외배송 -설치기사 -무상설치 -렌탈 -렌탈비 -정수기 -공기청정기 -안마의자 -AS -리퍼제품 -가전패키지')
    keywords = [f"{main} +{clean} {eliminate_ads_keyword}" for main in main_keywords for clean in clean_key]
    crawled_data = []
    for key in keywords:
        logging.info(f"[{key}] 키워드 검색 중")
        st.write(f"#### 키워드 '{key}' 크롤링 시작...")
        crawled_count_for_key = 0
        url = f'https://search.naver.com/search.naver?ssc=tab.cafe.all&sm=tab_jum&query={key.replace(" ", "+")}'
        driver.get(url)
        driver.implicitly_wait(15)
        time.sleep(1)
        breaker = False
        start_n, end_n = 1, 21
        while not breaker and crawled_count_for_key < max_items_per_keyword:
            check_soup = BeautifulSoup(driver.page_source, 'lxml')
            check_len = len(check_soup.select('div.title_area'))
            if check_len == 0 and start_n == 1:
                st.warning(f"'{key}'에 대한 검색 결과가 없습니다.")
                break
            for i in range(start_n, end_n):
                if crawled_count_for_key >= max_items_per_keyword:
                    breaker = True
                    break
                if i > check_len:
                    breaker = True
                    break
                xpath = f'//*[@id="main_pack"]/section/div[1]/ul/li[{i}]/div/div[2]/div[2]/a'
                try:
                    site = driver.find_element(By.XPATH, xpath)
                    href_value = site.get_attribute("href")
                    if not any(element in str(href_value) for element in eliminate_cafe):
                        site.click()
                        time.sleep(random.uniform(0.5, 1))
                        driver.switch_to.window(driver.window_handles[1])
                        title = ''
                        for _ in range(5):
                            try:
                                driver.switch_to.frame('cafe_main')
                                soup = BeautifulSoup(driver.page_source, 'lxml')
                                title = soup.select_one('h3.title_text').text
                                if title: break
                            except:
                                driver.refresh()
                                time.sleep(1)
                        if len(title) != 0:
                            contents = [c.text for c in soup.select('div.se-component-content, div.ContentRenderer')]
                            contents = ' '.join(contents).replace('\u200b', ' ').strip()
                            date = soup.select_one('span.date').text
                            comments = [c.text for c in soup.select('span.text_comment')]
                            crawled_data.append({
                                "keyword": key, "date": date.strip(), "title": title.strip(), "content": contents,
                                "comments": ", ".join(comments), "site": "네이버카페", "url": driver.current_url
                            })
                            crawled_count_for_key += 1
                            st.text(f"  - [{crawled_count_for_key}/{max_items_per_keyword}] {title.strip()[:40]}...")
                except Exception:
                    pass
                finally:
                    if len(driver.window_handles) > 1:
                        driver.close()
                    driver.switch_to.window(driver.window_handles[0])
            if breaker: break
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            start_n += 20
            end_n += 20
    driver.quit()
    st.success("모든 키워드에 대한 크롤링이 완료되었습니다.")
    return pd.DataFrame(crawled_data).drop_duplicates(subset='url', keep='first').reset_index(drop=True)

def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\\n', ' ', text)
        text = re.sub(r'\\s+', ' ', text).strip()
    return text

def filter_ads_by_title(df):
    # .ipynb과 app.py의 삭제 키워드를 통합하여 리스트 확장
    words_to_delete = [
        # app.py 기존 키워드
        '협찬', '체험단', '무료체험', '예약링크', '홍보','무료제공','무상제공','위스토어', '출석체크','긴급','가입인사','모집','양도매매','판매',
        '산후조리원 후기', '산후조리원후기', '임테기', '레시피', '출산후기',
        # .ipynb 추가 키워드 (중복 제외)
        '이벤트'
    ]
    pattern = '|'.join(words_to_delete)
    df['title'] = df['title'].astype(str)
    mask = df['title'].str.contains(pattern, case=False, na=False)
    return df[~mask]

def get_keyword_lists():
    return { "person": ["나는", "제가", "저는", "우리", "남편", "아내", "아이", "딸", "아들", "힘들", "고민", "좋아하", "싫어하", "사랑", "짜증", "속상", "행복", "걱정", "눈물"], "fiction": ["그는", "그녀는", "되뇌었다", "속삭였다", "눈물이", "바라보며", "기억했다", "미소를 지었다", "한참을", "창밖을", "어둠", "사라졌다", "느껴졌다"], "religious": ["하나님", "예수", "성령", "회개", "지옥", "천국", "복음", "기도", "전도", "영혼", "창조", "죄", "구원", "용서", "믿음", "십자가", "사탄", "말씀", "믿으십시오", "사람은 죄인입니다", "말세", "마귀", "영적", "부활", "기독교"], "ad": ["상품", "옵션", "ml", "세트", "강화유리", "액정보호필름", "보호필름", "최저가격", "적립", "배송"], "system_messages": ["문제가 발생했습니다", "다시 시도해주세요", "페이지를 찾을 수 없습니다", "시스템 오류", "불편을 드려 죄송합니다", "광고 후 계속됩니다", "다음 동영상", "subject author"], "url": ["http", "https", "naver", "com", "smartstore", "blog", "네이버쇼핑", "브랜드", "공식몰", "kr", "인스타그램"], "ad_descriptive": ["장점", "단점", "기능", "특징", "효과적", "효율적", "성능", "방수", "편안함", "내구성", "디자인", "착용감", "추천", "최적", "우수", "제공", "즐길 수 있습니다", "완벽한", "보장", "탁월한"], "msg_request": ["쪽지", "쪽찌", "보냈", "보낼께", "부탁드려", "성함", "업체", "정보", "알려주", "알수있"], "encyclopedic": ["설명", "의미", "정의", "특징", "성분", "화학식", "연구", "사용", "효능", "발견", "구성", "기원", "기술", "재료", "원리", "효과", "명칭", "개념", "유래"] }

def apply_rule_based_labeling(df_flat):
    keywords = get_keyword_lists()
    df_flat['label'] = 1
    progress_bar = st.progress(0, text="Rule 기반 필터링 적용 중...")
    total_rows = len(df_flat)
    for i, (index, row) in enumerate(df_flat.iterrows()):
        sentence = str(row['sentence']).lower()
        if any(kw in sentence for kw in keywords['ad']): df_flat.loc[index, 'label'] = 0
        elif any(kw in sentence for kw in keywords['system_messages']): df_flat.loc[index, 'label'] = 5
        elif any(kw in sentence for kw in keywords['url']): df_flat.loc[index, 'label'] = 6
        elif sum(kw in sentence for kw in keywords['ad_descriptive']) >= 3: df_flat.loc[index, 'label'] = 7
        elif sum(kw in sentence for kw in keywords['msg_request']) >= 2: df_flat.loc[index, 'label'] = 8
        elif sum(kw in sentence for kw in keywords['encyclopedic']) >= 4: df_flat.loc[index, 'label'] = 9
        if total_rows > 0:
            progress_bar.progress((i + 1) / total_rows)
    for doc_id, group in df_flat.groupby('doc_id'):
        fiction_score = sum(any(kw in str(s).lower() for kw in keywords['fiction']) for s in group['sentence'])
        religious_score = sum(any(kw in str(s).lower() for kw in keywords['religious']) for s in group['sentence'])
        if fiction_score >= 4: df_flat.loc[group.index, 'label'] = 2
        elif religious_score >= 5: df_flat.loc[group.index, 'label'] = 4
    progress_bar.empty()
    return df_flat

def expand_keywords_with_ai(api_key, seed_keyword, product_category, num_to_generate):
    st.info(f"**AI 키워드 확장**: `{seed_keyword}` (제품군: `{product_category or '미지정'}`) 키워드를 `{num_to_generate}`개의 핵심 단어로 확장합니다.")
    prompt = f"""
[역할]
당신은 특정 단어의 핵심 의미를 파악하고, 그와 관련된 다양한 표현을 찾아내는 전문 카피라이터입니다. 당신의 임무는 긴 검색어가 아닌, 대체 가능한 '하나의 단어'를 찾는 것입니다.

[작업]
주어진 'Seed 키워드'와 '제품군'을 바탕으로, 일반 사용자들이 해당 키워드 대신 사용할만한 유의어, 동의어, 혹은 구어체 표현을 찾아주세요.

[핵심 조건]
1. 결과는 반드시 **한 단어 형태의 명사**여야 합니다. (예: '스타일러 사용법'은 안됨, '의류관리기'는 가능)
2. 생성된 목록에는 반드시 원래의 'Seed 키워드'가 포함되어야 합니다.
3. 총 {num_to_generate}개의 키워드를 생성해주세요.

[입력]
- Seed 키워드: '{seed_keyword}'
- 제품군: '{product_category}'

[출력 형식]
- 오직 쉼표(,)로만 구분된 텍스트.
- 번호, 설명, 줄바꿈 등 다른 어떤 문자도 포함하지 마세요.
- 예시: 스타일러,의류관리기,옷관리기
"""
    try:
        client = openai.OpenAI(api_key=api_key)
        with st.spinner("AI가 핵심 동의어를 생성하고 있습니다..."):
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=256
            )
            ai_generated_text = response.choices[0].message.content.strip()
            expanded_keywords = [kw.strip() for kw in ai_generated_text.split(',') if kw.strip()]

        if not expanded_keywords:
            st.warning("AI가 키워드를 생성하지 못했습니다. 기본 키워드만 사용합니다.")
            return [seed_keyword]

        final_keywords = [seed_keyword]
        for kw in expanded_keywords:
            if kw not in final_keywords:
                final_keywords.append(kw)
        
        return final_keywords[:num_to_generate]

    except Exception as e:
        st.error(f"AI API 호출 오류: {e}")
        st.warning("오류로 인해 기본 키워드만 사용합니다.")
        return [seed_keyword]

def prepare_and_upload_to_qdrant(df, host, port, collection_name, batch_size=100):
    if df.empty:
        st.warning("업로드할 데이터가 없습니다.")
        return
    if 'sentence_nouns' not in df.columns:
        st.error("'sentence_nouns' 컬럼이 없어 Qdrant 업로드를 진행할 수 없습니다. 이전 단계를 확인해주세요.")
        return

    try:
        nlp_models = load_nlp_models()
        meaning_model = nlp_models["meaning"]
        topic_model = nlp_models["topic"]

        st.info(f"Qdrant 서버({host}:{port})에 연결합니다...")
        client = QdrantClient(host=host, port=port, timeout=60.0)

        if client.collection_exists(collection_name=collection_name):
            st.warning(f"기존 컬렉션 '{collection_name}'을(를) 삭제합니다.")
            client.delete_collection(collection_name=collection_name)
        
        st.info(f"'{collection_name}' 컬렉션을 새로 생성합니다.")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "meaning": VectorParams(size=meaning_model.get_sentence_embedding_dimension(), distance=Distance.COSINE),
                "topic": VectorParams(size=topic_model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
            }
        )
        st.success(f"✅ 컬렉션 준비 완료: {collection_name}")

        points_to_upload = []
        total_rows = len(df)
        progress_bar = st.progress(0, text="데이터 포인트 생성 및 업로드 진행 중...")
        
        for processed_count, (_, row) in enumerate(df.iterrows(), 1):
            sentence_text = str(row["sentence"]).strip()
            noun_text = str(row["sentence_nouns"]).strip()

            vectors = {
                "meaning": meaning_model.encode("query: " + sentence_text).tolist(),
                "topic": topic_model.encode(noun_text).tolist()
            }
            
            payload = {
                "sentence": sentence_text,
                "sentence_nouns": noun_text,
                "doc_id": int(row["doc_id"]),
                "sentence_idx": int(row["sentence_idx"])
            }
            try:
                date_obj = pd.to_datetime(row['date']).to_pydatetime()
                payload['date_timestamp'] = int(date_obj.timestamp())
            except: 
                payload['date_timestamp'] = -1
            
            point = PointStruct(id=str(uuid.uuid4()), vector=vectors, payload=payload)
            points_to_upload.append(point)
            
            if len(points_to_upload) >= batch_size:
                client.upsert(collection_name=collection_name, points=points_to_upload, wait=True)
                points_to_upload = []
            
            progress_bar.progress(processed_count / total_rows, text=f"데이터 포인트 생성 및 업로드 진행 중... ({processed_count}/{total_rows})")
        
        if points_to_upload:
            client.upsert(collection_name=collection_name, points=points_to_upload, wait=True)
        
        progress_bar.empty()
        st.success(f"🎉 총 {total_rows}개 데이터가 Qdrant '{collection_name}' 컬렉션에 성공적으로 업로드되었습니다.")
        st.balloons()

    except Exception as e:
        st.error(f"Qdrant 작업 중 오류가 발생했습니다: {e}")


# --- Streamlit UI 구성 ---
st.title("🚀 잠재고객 발굴 및 웹 데이터 처리 시스템")

with st.sidebar:
    st.header("⚙️ 시스템 설정")
    st.markdown("---")
    st.header("🗄️ Vector DB 설정")
    qdrant_host = st.text_input("Qdrant Host", "localhost", help="Qdrant 서버의 주소를 입력합니다.")
    qdrant_port = st.number_input("Qdrant Port", value=6333, help="Qdrant 서버의 포트를 입력합니다.")

    openai_api_key = st.text_input("OpenAI API Key", type="password", help="키워드 확장을 위해 OpenAI API 키를 입력해주세요.")
    seed_keyword = st.text_input("1. Seed 키워드", "스타일러", help="분석의 중심이 되는 핵심 키워드를 입력합니다.")
    product_category = st.selectbox("2. 제품군 (선택)", ["선택안함", "의류", "가전", "뷰티", "식품", "리빙"], index=2, help="키워드 확장의 방향성을 제시합니다.")
    if product_category == "선택안함": product_category = ""
    start_date, end_date = st.date_input("3. 수집 기간", [datetime.now() - timedelta(days=365*5), datetime.now()], help="데이터를 수집할 기간을 설정합니다.")
    num_expanded = st.slider("4. Seed 키워드 확장 갯수", 1, 10, 2, help="하나의 Seed 키워드를 몇 개의 연관 키워드로 확장할지 결정합니다.")
    num_crawls_per_keyword = st.slider("5. 키워드별 수집 게시물 수", 5, 50, 5, help="확장된 각 키워드마다 몇 개의 게시물을 수집할지 결정합니다.")
    
    start_processing = st.button("처리 시작", use_container_width=True)

if start_processing:
    if not openai_api_key: st.error("OpenAI API 키를 입력해주세요."); st.stop()
    
    # 단계 1: AI 키워드 확장
    with st.container(border=True):
        st.header("단계 1: AI 키워드 확장")
        main_keywords = expand_keywords_with_ai(openai_api_key, seed_keyword, product_category, num_expanded)
        st.table(pd.DataFrame(main_keywords, columns=["확장된 메인 키워드"]))

    # 단계 2: 웹 데이터 수집
    with st.container(border=True):
        st.header("단계 2: 웹 데이터 수집 (네이버 카페)")
        crawled_df = execute_cafe_crawl_procedure(
            main_keywords=main_keywords,
            clean_key=["옷"],
            max_items_per_keyword=num_crawls_per_keyword
        )
        if crawled_df.empty: st.error("수집된 데이터가 없습니다."); st.stop()
        st.subheader(f"📊 수집된 총 문서: {len(crawled_df)}개"); st.dataframe(crawled_df.head())
        st.session_state.crawled_df = crawled_df

    # 단계 3: 데이터 정제, 필터링 및 명사 추출
    with st.container(border=True):
        st.header("단계 3: 데이터 정제, 필터링 및 명사 추출")
        if 'crawled_df' in st.session_state and not st.session_state.crawled_df.empty:
            df_to_process = st.session_state.crawled_df.copy()
            
            # ⭐️ --- 노트북 로직 반영: 개선된 초기 정제 파이프라인 --- ⭐️
            st.subheader("3-1. 초기 정제 (.ipynb 로직 통합)")
            len_before = len(df_to_process)
            
            # 1. Null 값 및 중복 URL 제거
            df_to_process.dropna(subset=['url', 'content'], inplace=True)
            df_to_process.drop_duplicates(subset='url', keep='first', inplace=True)
            
            # 2. 날짜 형식 변환 및 기간 필터링
            df_to_process['date'] = pd.to_datetime(df_to_process['date'], errors='coerce')
            df_to_process.dropna(subset=['date'], inplace=True)
            df_to_process = df_to_process[(df_to_process['date'] >= pd.to_datetime(start_date)) & (df_to_process['date'] <= pd.to_datetime(end_date))]

            # 3. 제목 기반 광고 필터링 (통합 키워드 리스트 사용)
            df_to_process = filter_ads_by_title(df_to_process)
            
            # 4. 본문, 댓글 최소 길이 필터링
            df_to_process = df_to_process[df_to_process['content'].str.len() >= 10].copy()
            df_to_process['comments'] = filter_comments(df_to_process['comments'])
            
            # 5. 컬럼 통합 (title + content + comments)
            df_to_process['title'] = df_to_process['title'].fillna('')
            df_to_process['content'] = df_to_process['content'].fillna('')
            df_to_process['comments'] = df_to_process['comments'].fillna('')
            df_to_process['review'] = df_to_process['title'] + ' ' + df_to_process['content'] + ' ' + df_to_process['comments']
            
            # 6. 반복 문구 제거
            df_to_process['review'] = remove_repetitive_phrases(df_to_process['review'])
            
            # 7. 최종 텍스트 정제 (한글/영어 제외 모두 제거)
            df_to_process['review'] = apply_final_text_clean(df_to_process['review'])

            df_filtered = df_to_process.reset_index(drop=True)
            st.write(f".ipynb 기반 전체 정제 후: {len_before} -> {len(df_filtered)}개 문서")
            # ⭐️ --- 노트북 로직 반영 끝 --- ⭐️
            
            
            # 3-2. 문장 분리 및 데이터 구조 변환
            st.subheader("3-2. 문장 단위 변환")
            with st.spinner("모든 문서를 문장 단위로 분리하고 있습니다... (kss)"):
                df_filtered['sentences_clean'] = df_filtered['review'].apply(
                    lambda x: split_sentences(x) if pd.notna(x) else []
                )
            rows = []
            with st.spinner("데이터 구조를 문장 단위로 변환 중입니다..."):
                for doc_idx, row in tqdm(df_filtered.iterrows(), total=df_filtered.shape[0]):
                    sentences = row["sentences_clean"]
                    for idx, sentence in enumerate(sentences):
                        if len(sentence.strip()) >= 15:
                            rows.append({
                                "sentence": sentence,
                                "doc_id": doc_idx,
                                "sentence_idx": idx,
                                "date": row['date']
                            })
            flat_df = pd.DataFrame(rows)
            st.write(f"총 {len(flat_df)}개의 유효 문장으로 변환되었습니다.")
            st.dataframe(flat_df.head())

            # 3-3. Rule 기반 1차 필터링
            st.subheader("3-3. Rule 기반 1차 필터링")
            labeled_df = apply_rule_based_labeling(flat_df)
            st.write("Rule 기반 1차 라벨링 완료.")
            labels_to_remove = [0, 2, 4, 5, 6, 7, 8, 9]
            final_df_before_model = labeled_df[~labeled_df['label'].isin(labels_to_remove)].copy()
            st.write(f"Rule 기반 필터링 후: {len(labeled_df)} -> {len(final_df_before_model)}개 문장")
            
            # 3-4. AI 모델 기반 2차 필터링
            st.subheader("3-4. AI 모델 기반 2차 필터링")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.write(f"(사용 디바이스: {device})")
            
            inference_model = load_inference_model('best_model.pt', device)
            
            if inference_model:
                nlp_models = load_nlp_models()
                embedding_model = nlp_models["topic"]
                
                inference_dataset = InferenceDataset(final_df_before_model, embedding_model)
                predictions = predict_with_model(inference_model, inference_dataset, device)
                
                final_df_before_model['predicted_label'] = predictions
                
                final_processed_df = final_df_before_model[final_df_before_model['predicted_label'] == 1].copy()
                final_processed_df = final_processed_df.drop(columns=['predicted_label', 'label'])
                st.success(f"AI 모델 필터링 후 최종 **{len(final_processed_df)}개**의 유효 문장을 추출했습니다.")
                
                st.subheader("3-5. 최종 데이터 명사 추출")
                with st.spinner("최종 문장에서 토픽 분석용 명사를 추출합니다..."):
                    okt_tagger = nlp_models["okt"]
                    final_processed_df['sentence_nouns'] = final_processed_df['sentence'].apply(lambda text: extract_nouns(text, okt_tagger))
                
                st.write("명사 추출 완료. 'sentence_nouns' 컬럼이 추가되었습니다.")
                st.dataframe(final_processed_df.head())
                st.session_state.final_df = final_processed_df
            else:
                st.error("모델 로드 실패로 2차 필터링을 건너뜁니다.")
                st.session_state.final_df = final_df_before_model
        else:
            st.warning("크롤링된 데이터가 없어 정제를 진행할 수 없습니다.")

    # 단계 4: Vector DB 업로드
    with st.container(border=True):
        st.header("단계 4: Vector DB 업로드")
        if 'final_df' in st.session_state and not st.session_state.final_df.empty:
            prepare_and_upload_to_qdrant(
                df=st.session_state.final_df,
                host=qdrant_host,
                port=qdrant_port,
                collection_name="sample_web"
            )
        else:
            st.warning("처리할 데이터가 없어 업로드를 진행할 수 없습니다.")