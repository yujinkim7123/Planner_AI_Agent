# 3_processors.py
import pandas as pd
import re
from kss import split_sentences
from tqdm import tqdm
import torch

class BaseProcessor:
    """모든 프로세서의 기반 클래스. 공통 처리 로직을 포함합니다."""
    def __init__(self, nlp_models, qdrant_manager):
        self.nlp = nlp_models
        self.qdrant = qdrant_manager

    def initial_clean(self, df):
        """데이터 소스별로 구현될 초기 정제 단계."""
        raise NotImplementedError("초기 정제 메소드를 구현해야 합니다.")

    def _split_to_sentences(self, df):
        """정제된 'review' 컬럼을 문장 단위로 분할합니다."""
        rows = []
        tqdm.pandas(desc="문장 분리")
        
        # 'review' 컬럼이 없는 경우를 대비
        if 'review' not in df.columns:
            print("'review' 컬럼이 없어 문장 분리를 건너뜁니다.")
            return pd.DataFrame()

        df['sentences'] = df['review'].progress_apply(lambda x: split_sentences(x) if pd.notna(x) else [])
        
        for doc_idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="문장 데이터 구조 변환"):
            for sent_idx, sentence in enumerate(row["sentences"]):
                if len(sentence.strip()) >= 15:
                    rows.append({
                        "sentence": sentence,
                        "doc_id": doc_idx,
                        "sentence_idx": sent_idx,
                        "date": row['date'],
                        "url": row['url']
                    })
        return pd.DataFrame(rows)

    def _filter_by_rules(self, df):
        """규칙 기반으로 광고, 소설 등 불필요한 문장을 필터링합니다."""
        keywords = self.nlp.get_rule_based_keywords()
        # 이 함수는 app.py의 apply_rule_based_labeling와 유사하게 구현되어야 함
        # 여기서는 간단히 광고 키워드만 체크하는 예시를 보입니다.
        ad_pattern = '|'.join(keywords['ad'] + keywords['ad_descriptive'] + keywords['url'])
        mask = df['sentence'].str.contains(ad_pattern, case=False, na=False)
        return df[~mask]

    def _filter_by_ai_model(self, df):
        """AI 모델을 사용하여 광고성 문장을 최종 필터링합니다."""
        if df.empty: return df
        dataset = self.nlp.create_inference_dataset(df)
        predictions = self.nlp.predict(dataset)
        df['prediction'] = predictions
        return df[df['prediction'] == 1].drop(columns=['prediction'])

    def _extract_nouns(self, df):
        """최종 문장에서 명사를 추출합니다."""
        tqdm.pandas(desc="명사 추출")
        df['sentence_nouns'] = df['sentence'].progress_apply(self.nlp.extract_nouns)
        return df

    def run_pipeline(self, df, collection_name):
        """전체 데이터 처리 파이프라인을 실행합니다."""
        print("1. 초기 데이터 정제를 시작합니다...")
        cleaned_df = self.initial_clean(df)
        
        print("2. 문장 단위로 분리합니다...")
        sentences_df = self._split_to_sentences(cleaned_df)
        
        print("3. 규칙 기반 필터링을 적용합니다...")
        rule_filtered_df = self._filter_by_rules(sentences_df)
        
        print("4. AI 모델 기반 필터링을 적용합니다...")
        ai_filtered_df = self._filter_by_ai_model(rule_filtered_df)
        
        print("5. 최종 데이터에서 명사를 추출합니다...")
        final_df = self._extract_nouns(ai_filtered_df)
        
        print(f"최종 처리된 문장 수: {len(final_df)}개")
        
        print("6. Vector DB에 업로드를 시작합니다...")
        self.qdrant.upload_data(final_df, collection_name)
        
        return final_df


class PortalBlogProcessor(BaseProcessor):
    """네이버 포털 및 블로그 데이터 전용 전처리기."""
    def initial_clean(self, df):
        # 날짜 형식 통일
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df.dropna(subset=['date', 'title', 'contents'], inplace=True)
        
        # 텍스트 정제
        for col in ['title', 'contents', 'comments']:
            if col in df.columns:
                df[col] = df[col].str.replace(r'<[^>]+>', '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # 광고성 제목 제거
        ad_words = ['협찬', '체험단', '무료체험', '예약링크', '홍보', '무료제공', '판매']
        pattern = '|'.join(ad_words)
        df = df[~df['title'].str.contains(pattern, case=False, na=False)]
        
        # 10글자 미만 본문 제거
        df = df[df['contents'].str.len() >= 10]
        
        # 'review' 컬럼 생성
        df['review'] = df['title'] + ' ' + df['contents'] + ' ' + df['comments'].fillna('')
        
        # 반복 문구 제거
        phrases_to_remove = ["쪽지드렸습니다", "안녕하세요", "감사합니다"]
        for phrase in phrases_to_remove:
            df['review'] = df['review'].str.replace(phrase, "", regex=False)
        
        return df.reset_index(drop=True)

class CafeProcessor(BaseProcessor):
    """네이버 카페 데이터 전용 전처리기."""
    def initial_clean(self, df):
        # 날짜 형식 통일
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df.dropna(subset=['date', 'title', 'contents'], inplace=True)

        # 텍스트 정제
        for col in ['title', 'contents', 'comments', 'board_titles']:
             if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'<[^>]+>', '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # 광고성 제목 제거
        ad_words = ['협찬', '체험단', '무료체험', '판매', '이벤트 후기', '벼룩']
        pattern = '|'.join(ad_words)
        df = df[~df['title'].str.contains(pattern, case=False, na=False)]
        
        # 게시판 이름 기준 필터링
        if 'board_titles' in df.columns:
            exclude_boards = ['이벤트 후기', '알뜰 벼룩', '쇼핑할인', '공구']
            board_pattern = '|'.join(exclude_boards)
            df = df[~df['board_titles'].str.contains(board_pattern, case=False, na=False)]

        # 댓글 처리 (10글자 미만 제거)
        df['comments'] = df['comments'].apply(lambda x: ' '.join([c for c in x.split('|') if len(c.strip()) >= 10]))

        # 10글자 미만 본문 제거
        df = df[df['contents'].str.len() >= 10]

        # 'review' 컬럼 생성
        df['review'] = df['title'] + ' ' + df['contents'] + ' ' + df['comments'].fillna('')

        # 반복 문구 제거
        phrases_to_remove = ["쪽지드렸습니다", "안녕하세요", "가입인사", "잘부탁드립니다"]
        for phrase in phrases_to_remove:
            df['review'] = df['review'].str.replace(phrase, "", regex=False)

        return df.reset_index(drop=True)