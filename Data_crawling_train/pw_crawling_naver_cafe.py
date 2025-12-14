import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup as bs
import pandas as pd
import time
import random
import logging
import re

# 네이버 로그인 정보
my_id = 'vorutesv'  # 네이버 아이디
my_pw = 'kimhs2925!#'  # 네이버 비밀번호

# 크롤링할 키워드와 카페 URL
keyword = ['중고']
cafe_url = 'https://cafe.naver.com/robotclear'

# 검색 파라미터
search_start_date = '2020-01-01'    # 검색 시작 날짜
search_end_date   = '2023-09-12'      # 검색 종료 날짜
max_pages = 2  # 최대 페이지 수 제한

# 데이터 저장용 리스트
titles = []
reviews = []
comments = []
dates = []
search_keywords = []
board_titles = []
urls = []

# 날짜 추출 함수
def extract_date(soup):
    try:
        date_element = soup.select_one('span.date')
        if date_element:
            date_text = date_element.text.strip()
            if "시간 전" in date_text or "분 전" in date_text or "방금" in date_text:
                return time.strftime('%Y-%m-%d', time.localtime())
            elif "." in date_text:
                return date_text.replace(".", "-")
        return "날짜 없음"
    except Exception as e:
        logging.error(f"날짜 추출 오류: {e}")
        return "날짜 없음"


# 본문 추출 함수
def extract_content(soup, include_images=True):
    contents = []
    selectors = [
        'div.se-module.se-module-text',
        'div.ContentRenderer',
        'div.scrap_added',
    ]
    for selector in selectors:
        elements = soup.select(selector)
        if elements:
            contents.extend([element.text.strip() for element in elements])

    if include_images:
        images = soup.select('img')
        for img in images:
            if img.get('src'):
                contents.append(f"[이미지: {img['src']}]")

    return ' '.join(contents).strip() if contents else "본문 없음"


# Playwright 크롤링 함수
async def scrape_cafe():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        # 네이버 로그인
        await page.goto('https://nid.naver.com/nidlogin.login')
        await page.fill('input[name="id"]', my_id)
        await page.fill('input[name="pw"]', my_pw)
        await page.click('button#log\\.login')  # 수정: \\ 사용
        await page.wait_for_timeout(3000)

        # 카페 메인 페이지로 이동
        await page.goto(cafe_url)
        await page.wait_for_timeout(2000)

        for k in keyword:
            print("%s 키워드 크롤링 중입니다" % k)

            # 검색
            search_box = await page.wait_for_selector('#topLayerQueryInput')
            await search_box.fill(k)
            await search_box.press('Enter')
            await page.wait_for_timeout(2000)

            # iframe 전환
            iframe = page.frame(name="cafe_main")
            if not iframe:
                print("iframe not found")
                continue

            # 데이터 수집 기간 설정
            await iframe.click('#currentSearchDateTop')
            await iframe.fill('#input_1_top', search_start_date)
            await iframe.fill('#input_2_top', search_end_date)
            await iframe.click('#btn_set_top')
            await page.wait_for_timeout(1000)

            # 검색 버튼 클릭
            try:
                await iframe.wait_for_selector('button.btn-search-green', timeout=30000)
                await iframe.click('button.btn-search-green')
                print("검색 버튼 클릭 성공")
            except Exception as e:
                print("검색 버튼 클릭 실패 %s" % e)
                continue

            await page.wait_for_timeout(3000)

            # 페이지 순회하며 게시글 URL 수집
            current_page_url = ""
            current_page = 0

            while current_page <= max_pages:
                # iframe 전환
                iframe = page.frame(name="cafe_main")  # 'cafe_main'이라는 이름의 iframe으로 전환
                if not iframe:
                    print("iframe 'cafe_main' not found")
                else:
                    # 'a.on' 요소 선택
                    current_page_element = await iframe.query_selector('a.on')
                    if current_page_element:
                        current_page_url = await current_page_element.get_attribute('href')
                        current_page = int(re.search(r'\.page=(\d+)', current_page_url).group(1))
                        print("First page URL: %s" % current_page_url)
                    else:
                        print("Element with selector 'a.on' not found in iframe")

                # 게시글 URL 수집
                page_url_list = []
                try:
                    page_url_list = await iframe.eval_on_selector_all(
                        'a.article', 'elements => elements.map(el => el.href)'
                    )
                    print("수집된 게시글 URL 수: %d" % len(page_url_list))
                except Exception as e:
                    print("게시글 URL 수집 실패: %s" % e)
                    break

                print("Crawling Search Page %d" % current_page)

                # 게시글 상세 정보 수집
                for url in page_url_list:
                    try:
                        await page.goto(url)
                        await page.wait_for_timeout(2000)

                        # iframe 전환
                        iframe = page.frame(name="cafe_main")
                        if not iframe:
                            print("iframe not found for %s" % url)
                            continue

                        # BeautifulSoup으로 페이지 소스 파싱
                        source = await iframe.content()
                        soup = bs(source, 'html.parser')

                        # 제목 수집
                        title = soup.select_one('h3.title_text')
                        titles.append(title.text.strip() if title else "제목 없음")

                        # 날짜 수집
                        date = extract_date(soup)
                        dates.append(date)

                        # 본문 수집
                        content = extract_content(soup)
                        reviews.append(content)

                        # 댓글 수집
                        comment_elements = soup.select('span.text_comment')
                        comment_texts = [c.text.strip() for c in comment_elements]
                        comments.append(" ".join(comment_texts) if comment_texts else "댓글 없음")

                        # 키워드, 게시판 이름, URL 저장
                        search_keywords.append(k)
                        board_title = soup.select_one('div.ArticleTitle')
                        board_titles.append(board_title.text.strip() if board_title else "게시판 없음")
                        urls.append(url)

                    except Exception as e:
                        print("Error processing %s: %s" % (url, e))
                        continue

                # Playwright에서 페이지 이동 및 iframe 전환
                current_page = current_page+1
                print(current_page)
                next_page_url = cafe_url + current_page_url[:-1] + str(current_page)  # 다음 페이지 URL 생성
                print("next page URL: ", next_page_url)
                await page.goto(next_page_url)  # 페이지 이동
                await page.wait_for_timeout(1000)  # 페이지 로드 대기

                # # iframe 전환
                # iframe = page.frame(name="cafe_main")  # 'cafe_main'이라는 이름의 iframe으로 전환
                # if not iframe:
                #     print("iframe 'cafe_main' not found")
                # else:
                #     print("iframe 'cafe_main' 전환 성공")

        await browser.close()

    # 데이터프레임 생성 및 저장
    data = {
        'date': dates,
        'keyword': search_keywords,
        'title': titles,
        'contents': reviews,
        'comments': comments,
        'board_titles': board_titles,
        'url': urls,
    }
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset='url', keep='first').reset_index(drop=True)
    print(df)
    df.to_csv("카페명_카테고리명_키워드_pw.csv", index=False, encoding='utf-8-sig')

# 비동기 함수 실행
asyncio.run(scrape_cafe())