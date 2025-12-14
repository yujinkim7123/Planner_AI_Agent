import logging
import random
import asyncio
from urllib.parse import urlparse
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup as bs
import time
import pandas as pd
import re
import warnings

warnings.filterwarnings(action='ignore')

main_keywords = ['청소']

# 제외하고 싶은 카페홈 주소
eliminate_cafe = ['https://cafe.naver.com/clubng',
                'https://cafe.naver.com/outletshop',
                'https://cafe.naver.com/gjtnwls123',
                'https://cafe.naver.com/movie567',
                'https://cafe.naver.com/komandos',
                'https://cafe.naver.com/comicity',
                'https://cafe.naver.com/bk1009',
                'https://cafe.naver.com/bestofmobile',
                'https://cafe.naver.com/illnesses',
                'https://cafe.naver.com/simsll',
                'https://cafe.naver.com/nexonsfw',
                'https://cafe.naver.com/ipod5',
                'https://cafe.naver.com/goldenpeach',
                'https://cafe.naver.com/itop5',
                'https://cafe.naver.com/peopledisc',
                'https://cafe.naver.com/soho',
                'https://cafe.naver.com/tictoc',
                'https://cafe.naver.com/s7942',
                'https://cafe.naver.com/simsll',
                'https://cafe.naver.com/godofhighschooljjang',
                'https://cafe.naver.com/joonggonara',
                'https://cafe.naver.com/hongdaeholic',
                'https://cafe.naver.com/poohstory',
                'https://cafe.naver.com/glycosarang',
                'https://cafe.naver.com/appleiphone']##추후에 크롤링 시도할 카페

pattern = '|'.join(eliminate_cafe)

####### 청소경험 키워드

#포털 광고 글 필터링 위함
#eliminate_ads_keyword='-광고 -특가 -할인 -세일 -수수료 -이벤트 -핫딜 -입주 -이사'
eliminate_ads_keyword = '-할인 -핫딜 -매도 -임대 -계약 -특가 -가이드 -업체 -입주 -이사'

#청소 관련 단어

clean_key =['경로'] #앞으로 해야함!

keywords=[]

for main in main_keywords:
    for clean in clean_key:
        keywords.append(main + '+%2B' + clean +' ' + eliminate_ads_keyword)

print(keywords)

import os
# 로그 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 데이터프레임 생성
dataframe = pd.DataFrame(columns=['date', 'keyword', 'title', 'contents', 'comments', 'site', 'url'])
index = 0


# async def main():
#     global index
#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=False)
#         context = await browser.new_context()
#         page = await context.new_page()

#         for key in keywords:
#             # logging.info(f"[{key}] 키워드 검색 중")
#             # url = f'https://search.naver.com/search.naver?ssc=tab.cafe.all&sm=tab_jum&query={key}'
#             # await page.goto(url)
#             # await page.wait_for_timeout(1000)

#             logging.info(f"[{key}] 키워드 검색 중")
#             url = f'https://search.naver.com/search.naver?ssc=tab.cafe.all&sm=tab_jum&query={key}'
#             await page.goto(url, wait_until="networkidle")  # 페이지 로딩 완료 대기

#             breaker = False
#             start_n, end_n = 1, 31

#             while not breaker:
#                 content = await page.content()
#                 soup = bs(content, 'lxml')
#                 check_len = len(soup.select('div.title_area'))

#                 for i in range(start_n, end_n):
#                     if i == check_len + 1:
#                         breaker = True
#                         break

#                     try:
#                         xpath = f'//*[@id="main_pack"]/section/div[1]/ul/li[{i}]/div/div[2]/div[2]/a'
#                         element = await page.query_selector(xpath)
#                         if not element:
#                             continue

#                         href_value = await element.get_attribute("href")
#                         if any(el in href_value for el in eliminate_cafe):
#                             continue

#                         await element.click()
#                         await page.wait_for_timeout(15000)

#                         # Switch to new tab
#                         new_page = await context.wait_for_event("page")
#                         await new_page.wait_for_load_state()

#                         # Extract data
#                         try:
#                             await new_page.frame_locator('iframe[name="cafe_main"]').wait_for()
#                             frame = new_page.frame(name="cafe_main")
#                             frame_content = await frame.content()
#                             frame_soup = bs(frame_content, 'lxml')

#                             title = frame_soup.select_one('h3.title_text').text
#                             contents = [i.text for i in frame_soup.select('div.se-component-content')]
#                             if not contents:
#                                 contents = [i.text for i in frame_soup.select('div.ContentRenderer')]
#                             contents = ' '.join(contents).replace('\u200b', ' ').strip()
#                             contents = re.sub(r'\s+', ' ', contents)

#                             date = frame_soup.select_one('span.date').text
#                             comments = [i.text for i in frame_soup.select('span.text_comment')]
#                             url = new_page.url

#                             # Add to dataframe
#                             input_data = [date, key, title, contents, comments, '네이버포탈', url]
#                             dataframe.loc[index] = input_data
#                             index += 1
#                         except Exception as e:
#                             logging.error(f"Error extracting data: {e}")
#                         finally:
#                             await new_page.close()

#                     except Exception as e:
#                         logging.error(f"Error processing element: {e}")

#                 await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
#                 await page.wait_for_timeout(1000)
#                 start_n += 30
#                 end_n += 30

#         # Close browser
#         await browser.close()

#     # Save to CSV
#     dataframe.drop_duplicates(subset='url', keep='first', inplace=True)
#     csv_filename = "naver_potal_카테고리명_키워드명.csv"
#     dataframe.to_csv(csv_filename, encoding="utf-8-sig")
#     logging.info(f"데이터가 '{csv_filename}' 파일로 저장되었습니다.")


# async def main():
#     global index
#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=False)
#         context = await browser.new_context()
#         page = await context.new_page()

#         for key in keywords:
#             logging.info(f"[{key}] 키워드 검색 중")
#             url = f'https://search.naver.com/search.naver?ssc=tab.cafe.all&sm=tab_jum&query={key}'
#             await page.goto(url, wait_until="networkidle")  # 페이지 로딩 완료 대기

#             breaker = False
#             start_n, end_n = 1, 31

#             while not breaker:
#                 if page.is_closed():  # 페이지가 닫혔는지 확인
#                     logging.error("Page is already closed. Skipping this iteration.")
#                     break

#                 content = await page.content()
#                 soup = bs(content, 'lxml')
#                 check_len = len(soup.select('div.title_area'))

#                 for i in range(start_n, end_n):
#                     if i == check_len + 1:
#                         breaker = True
#                         break

#                     try:
#                         xpath = f'//*[@id="main_pack"]/section/div[1]/ul/li[{i}]/div/div[2]/div[2]/a'
#                         try:
#                             element = await page.query_selector(xpath)
#                             if not element:
#                                 continue
#                         except Exception as e:
#                             logging.error(f"Error querying selector: {e}")
#                             continue

#                         href_value = await element.get_attribute("href")
#                         if any(el in href_value for el in eliminate_cafe):
#                             continue

#                         await element.click()

#                         try:
#                             new_page = await context.wait_for_event("page", timeout=60000)
#                             await new_page.wait_for_load_state()

#                             # Extract data
#                             try:
#                                 await new_page.frame_locator('iframe[name="cafe_main"]').wait_for()
#                                 frame = new_page.frame(name="cafe_main")
#                                 frame_content = await frame.content()
#                                 frame_soup = bs(frame_content, 'lxml')

#                                 title = frame_soup.select_one('h3.title_text').text
#                                 contents = [i.text for i in frame_soup.select('div.se-component-content')]
#                                 if not contents:
#                                     contents = [i.text for i in frame_soup.select('div.ContentRenderer')]
#                                 contents = ' '.join(contents).replace('\u200b', ' ').strip()
#                                 contents = re.sub(r'\s+', ' ', contents)

#                                 date = frame_soup.select_one('span.date').text
#                                 comments = [i.text for i in frame_soup.select('span.text_comment')]
#                                 url = new_page.url

#                                 # Add to dataframe
#                                 input_data = [date, key, title, contents, comments, '네이버포탈', url]
#                                 dataframe.loc[index] = input_data
#                                 index += 1
#                             except Exception as e:
#                                 logging.error(f"Error extracting data: {e}")
#                             finally:
#                                 await new_page.close()
#                         except Exception as e:
#                             logging.error(f"Error waiting for new page: {e}")
#                             continue

#                     except Exception as e:
#                         logging.error(f"Error processing element: {e}")

#                 await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
#                 await page.wait_for_timeout(1000)
#                 start_n += 30
#                 end_n += 30

#         # Close browser
#         await browser.close()

#     # Save to CSV
#     dataframe.drop_duplicates(subset='url', keep='first', inplace=True)
#     csv_filename = "naver_potal_카테고리명_키워드명.csv"
#     dataframe.to_csv(csv_filename, encoding="utf-8-sig")
#     logging.info(f"데이터가 '{csv_filename}' 파일로 저장되었습니다.")


# async def main():
#     global index
#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=False)
#         context = await browser.new_context()
#         page = await context.new_page()

#         for key in keywords:
#             logging.info(f"[{key}] 키워드 검색 중")
#             url = f'https://search.naver.com/search.naver?ssc=tab.cafe.all&sm=tab_jum&query={key}'
#             await page.goto(url, wait_until="networkidle")  # 페이지 로딩 완료 대기

#             breaker = False
#             # start_n, end_n = 1, 31
#             start_n, end_n = 1, 2  #  for test

#             while not breaker:
#                 if page.is_closed():  # 페이지가 닫혔는지 확인
#                     logging.error("Page is already closed. Skipping this iteration.")
#                     break

#                 content = await page.content()
#                 soup = bs(content, 'lxml')
#                 check_len = len(soup.select('div.title_area'))

#                 for i in range(start_n, end_n):
#                     if i == check_len + 1:
#                         breaker = True
#                         break

#                     try:
#                         xpath = f'//*[@id="main_pack"]/section/div[1]/ul/li[{i}]/div/div[2]/div[2]/a'
#                         try:
#                             element = await page.query_selector(xpath)
#                             if not element:
#                                 continue
#                         except Exception as e:
#                             logging.error(f"Error querying selector: {e}")
#                             continue

#                         href_value = await element.get_attribute("href")
#                         if any(el in href_value for el in eliminate_cafe):
#                             continue

#                         await element.click()

#                         try:
#                             new_page = await context.wait_for_event("page", timeout=60000)
#                             await new_page.wait_for_load_state()

#                             # Extract data
#                             try:
#                                 # Frame 처리 수정
#                                 frame_locator = new_page.frame_locator('iframe[name="cafe_main"]')
#                                 await frame_locator.locator('body').wait_for()  # 프레임 내 로딩 대기
#                                 frame = new_page.frame(name="cafe_main")
#                                 frame_content = await frame.content()
#                                 frame_soup = bs(frame_content, 'lxml')

#                                 title = frame_soup.select_one('h3.title_text').text
#                                 contents = [i.text for i in frame_soup.select('div.se-component-content')]
#                                 if not contents:
#                                     contents = [i.text for i in frame_soup.select('div.ContentRenderer')]
#                                 contents = ' '.join(contents).replace('\u200b', ' ').strip()
#                                 contents = re.sub(r'\s+', ' ', contents)

#                                 date = frame_soup.select_one('span.date').text
#                                 comments = [i.text for i in frame_soup.select('span.text_comment')]
#                                 url = new_page.url

#                                 # Add to dataframe
#                                 input_data = [date, key, title, contents, comments, '네이버포탈', url]
#                                 dataframe.loc[index] = input_data
#                                 index += 1
#                             except Exception as e:
#                                 logging.error(f"Error extracting data: {e}")
#                             finally:
#                                 await new_page.close()
#                         except Exception as e:
#                             logging.error(f"Error waiting for new page: {e}")
#                             continue

#                     except Exception as e:
#                         logging.error(f"Error processing element: {e}")

#                 await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
#                 await page.wait_for_timeout(1000)
#                 start_n += 30
#                 end_n += 30

#         # Close browser
#         await browser.close()

#     # Save to CSV
#     dataframe.drop_duplicates(subset='url', keep='first', inplace=True)
#     csv_filename = "naver_potal_카테고리명_키워드명.csv"
#     dataframe.to_csv(csv_filename, encoding="utf-8-sig")
#     logging.info(f"데이터가 '{csv_filename}' 파일로 저장되었습니다.")


from tqdm.asyncio import tqdm as tqdm_async  # tqdm의 비동기 버전 사용

async def main():
    global index
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        # 키워드 진행 상황 표시
        for key in tqdm_async(keywords, desc="키워드 진행"):
            logging.info(f"[{key}] 키워드 검색 중")
            url = f'https://search.naver.com/search.naver?ssc=tab.cafe.all&sm=tab_jum&query={key}'
            await page.goto(url, wait_until="networkidle")  # 페이지 로딩 완료 대기

            breaker = False
            start_n, end_n = 1, 2  # for test

            # 게시물 진행 상황 표시
            with tqdm_async(total=30, desc="게시물 진행", leave=False) as progress:
                while not breaker:
                    if page.is_closed():  # 페이지가 닫혔는지 확인
                        logging.error("Page is already closed. Skipping this iteration.")
                        break

                    content = await page.content()
                    soup = bs(content, 'lxml')
                    check_len = len(soup.select('div.title_area'))

                    for i in range(start_n, end_n):
                        if i == check_len + 1:
                            breaker = True
                            break

                        try:
                            xpath = f'//*[@id="main_pack"]/section/div[1]/ul/li[{i}]/div/div[2]/div[2]/a'
                            try:
                                element = await page.query_selector(xpath)
                                if not element:
                                    continue
                            except Exception as e:
                                logging.error(f"Error querying selector: {e}")
                                continue

                            href_value = await element.get_attribute("href")
                            if any(el in href_value for el in eliminate_cafe):
                                continue

                            await element.click()

                            try:
                                new_page = await context.wait_for_event("page", timeout=60000)
                                await new_page.wait_for_load_state()

                                # Extract data
                                try:
                                    # Frame 처리 수정
                                    frame_locator = new_page.frame_locator('iframe[name="cafe_main"]')
                                    await frame_locator.locator('body').wait_for()  # 프레임 내 로딩 대기
                                    frame = new_page.frame(name="cafe_main")
                                    frame_content = await frame.content()
                                    frame_soup = bs(frame_content, 'lxml')

                                    title = frame_soup.select_one('h3.title_text').text
                                    contents = [i.text for i in frame_soup.select('div.se-component-content')]
                                    if not contents:
                                        contents = [i.text for i in frame_soup.select('div.ContentRenderer')]
                                    contents = ' '.join(contents).replace('\u200b', ' ').strip()
                                    contents = re.sub(r'\s+', ' ', contents)

                                    date = frame_soup.select_one('span.date').text
                                    comments = [i.text for i in frame_soup.select('span.text_comment')]
                                    url = new_page.url

                                    # Add to dataframe
                                    input_data = [date, key, title, contents, comments, '네이버포탈', url]
                                    dataframe.loc[index] = input_data
                                    index += 1

                                    # Update progress
                                    progress.update(1)
                                except Exception as e:
                                    logging.error(f"Error extracting data: {e}")
                                finally:
                                    await new_page.close()
                            except Exception as e:
                                logging.error(f"Error waiting for new page: {e}")
                                continue

                        except Exception as e:
                            logging.error(f"Error processing element: {e}")

                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                    await page.wait_for_timeout(1000)
                    start_n += 30
                    end_n += 30

        # Close browser
        await browser.close()

    # Save to CSV
    dataframe.drop_duplicates(subset='url', keep='first', inplace=True)
    csv_filename = "naver_potal_카테고리명_키워드명_pw.csv"
    dataframe.to_csv(csv_filename, encoding="utf-8-sig")
    logging.info(f"데이터가 '{csv_filename}' 파일로 저장되었습니다.")


# Run the script
asyncio.run(main())