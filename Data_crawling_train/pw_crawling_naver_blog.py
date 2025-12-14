import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import re
import pandas as pd
import urllib.parse
import urllib.request
import json
import time

# Naver API key 입력
client_id = 'E8L9EFxRYV1RtExrec38'
client_secret = 'PA0IZ_XEEK'

# 검색어 입력
keyword = "세탁기"
encText = urllib.parse.quote(keyword)

# 검색을 끝낼 페이지 입력
end = 2
if end == "":
    end = 1
else:
    end = int(end)
print("\n 1 ~ ", end, "페이지 까지 크롤링을 진행 합니다")

urls_list = []
postdate = []
titles = []

# 네이버 블로그 API 호출
for start in range(end):
    url = "https://openapi.naver.com/v1/search/blog?query=" + encText + "&start=" + str((start * 10) + 1)
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if rescode == 200:
        response_body = response.read()
        data = json.loads(response_body.decode('utf-8'))['items']
        for row in data:
            if 'blog.naver' in row['link']:
                urls_list.append(row['link'])
                postdate.append(row['postdate'])
                title = row['title']
                # html 태그 제거
                pattern1 = '<[^>]*>'
                title = re.sub(pattern=pattern1, repl='', string=title)
                # print(title)
                titles.append(title)
        time.sleep(2)
    else:
        print("Error Code:" + rescode)


async def scrape_blog():
    contents = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        for i in urls_list:
            print(f"Processing URL: {i}")
            try:
                page = await context.new_page()
                # 타임아웃을 60초로 설정하고 페이지 로딩 대기
                await page.goto(i, wait_until="load", timeout=60000)

                # iframe 탐색 및 전환
                iframe = page.frame(name="mainFrame")
                if not iframe:
                    print(f"iframe not found for {i}")
                    contents.append("iframe not found")
                    continue

                # iframe 내부 콘텐츠 로딩 대기
                try:
                    # await iframe.wait_for_selector("div.se-main-container", timeout=30000)
                    await iframe.wait_for_selector("div.se-main-container, div#postViewArea", timeout=30000)
                except Exception as e:
                    print(f"Content not found in iframe for {i}: {e}")
                    contents.append("content not found")
                    continue

                # iframe의 HTML 가져오기
                source = await iframe.evaluate("document.documentElement.outerHTML")
                html = BeautifulSoup(source, 'html.parser')

                # 기사 텍스트 가져오기
                content = html.select_one("div.se-main-container")
                if not content:
                    # 다른 선택자 확인
                    content = html.select_one("div#postViewArea")
                if not content:
                    print(f"Content not found for {i}")
                    contents.append("content not found")
                    continue

                # 텍스트 다듬기
                content = content.get_text(strip=True)
                contents.append(content)

                await page.close()
            except Exception as e:
                print(f"Error processing {i}: {e}")
                contents.append("error")

        await browser.close()

    # 데이터프레임 생성 및 저장
    blog_df = pd.DataFrame({'date': postdate, 'title': titles, 'content': contents, 'url': urls_list})
    blog_df.to_csv('blog_pw.csv', index=False, encoding='utf-8-sig')
    print(blog_df)


# 비동기 함수 실행
asyncio.run(scrape_blog())