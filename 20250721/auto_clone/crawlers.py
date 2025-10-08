# 2_crawlers.py
import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup as bs
import urllib.request
import json
import re
from tqdm.asyncio import tqdm

class BaseCrawler:
    """크롤러의 기반이 되는 클래스"""
    def __init__(self, keywords):
        self.keywords = keywords

    async def crawl(self):
        raise NotImplementedError("각 크롤러는 crawl 메소드를 구현해야 합니다.")

    def _save_to_df(self, data_list, columns):
        """수집된 데이터를 데이터프레임으로 변환하고 중복을 제거합니다."""
        if not data_list:
            return pd.DataFrame()
        df = pd.DataFrame(data_list, columns=columns)
        df.drop_duplicates(subset='url', keep='first', inplace=True)
        return df.reset_index(drop=True)

class PortalCrawler(BaseCrawler):
    """네이버 포털의 카페 검색 결과를 크롤링합니다."""
    async def crawl(self):
        data = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()

            for key in tqdm(self.keywords, desc="네이버 포털 크롤링"):
                search_url = f'https://search.naver.com/search.naver?ssc=tab.cafe.all&sm=tab_jum&query={key}'
                await page.goto(search_url, wait_until="networkidle")

                for _ in range(2): # 2페이지 스크롤 다운
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                    await asyncio.sleep(1)

                content = await page.content()
                soup = bs(content, 'lxml')
                
                posts = soup.select('ul.cafe_list li.cafe_item')
                for post in posts:
                    title_tag = post.select_one('a.title_link')
                    content_tag = post.select_one('p.dsc_txt')
                    url = title_tag['href'] if title_tag else ''
                    title = title_tag.text if title_tag else ''
                    contents = content_tag.text if content_tag else ''
                    date = post.select_one('span.sub_time').text if post.select_one('span.sub_time') else ''

                    data.append({
                        "date": date, "keyword": key, "title": title, 
                        "contents": contents, "comments": "", "site": "네이버포털", "url": url
                    })
            await browser.close()
        return self._save_to_df(data, ["date", "keyword", "title", "contents", "comments", "site", "url"])


class BlogCrawler(BaseCrawler):
    """네이버 블로그 API와 Playwright를 사용하여 블로그를 크롤링합니다."""
    def __init__(self, keywords, client_id, client_secret):
        super().__init__(keywords)
        self.client_id = client_id
        self.client_secret = client_secret
        self.urls_list = []
        self.postdates = []
        self.titles = []

    def _fetch_blog_urls(self):
        """네이버 API를 통해 블로그 게시물 URL 목록을 가져옵니다."""
        for keyword in self.keywords:
            enc_text = urllib.parse.quote(keyword)
            for start in range(1, 3):  # 2페이지 까지만 (1~20개)
                url = f"https://openapi.naver.com/v1/search/blog?query={enc_text}&start={start*10-9}&display=10"
                request = urllib.request.Request(url)
                request.add_header("X-Naver-Client-Id", self.client_id)
                request.add_header("X-Naver-Client-Secret", self.client_secret)
                response = urllib.request.urlopen(request)
                if response.getcode() == 200:
                    data = json.loads(response.read().decode('utf-8'))['items']
                    for row in data:
                        if 'blog.naver.com' in row['link']:
                            self.urls_list.append(row['link'])
                            self.postdates.append(row['postdate'])
                            self.titles.append(re.sub(r'<[^>]*>', '', row['title']))
                else:
                    print(f"Error Code: {response.getcode()}")

    async def crawl(self):
        self._fetch_blog_urls()
        contents_list = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            for url in tqdm(self.urls_list, desc="네이버 블로그 크롤링"):
                page = await context.new_page()
                try:
                    await page.goto(url, timeout=30000)
                    iframe = page.frame(name="mainFrame")
                    if iframe:
                        html = await iframe.content()
                        soup = bs(html, 'html.parser')
                        content_area = soup.select_one("div.se-main-container, div#postViewArea")
                        contents_list.append(content_area.get_text(strip=True) if content_area else "본문 없음")
                    else:
                        contents_list.append("메인 프레임 없음")
                except Exception as e:
                    print(f"블로그 크롤링 오류 ({url}): {e}")
                    contents_list.append("오류 발생")
                finally:
                    await page.close()
            await browser.close()
        
        data = {
            "date": self.postdates, "title": self.titles, 
            "contents": contents_list, "url": self.urls_list,
            "keyword": self.keywords[0] if self.keywords else "", "comments": "", "site": "네이버블로그"
        }
        return self._save_to_df(pd.DataFrame(data), ["date", "keyword", "title", "contents", "comments", "site", "url"])

class CafeCrawler(BaseCrawler):
    """특정 카페에 로그인하여 키워드로 게시글을 검색하고 크롤링합니다."""
    def __init__(self, keywords, cafe_url, naver_id, naver_pw, start_date, end_date, max_pages=2):
        super().__init__(keywords)
        self.cafe_url = cafe_url
        self.naver_id = naver_id
        self.naver_pw = naver_pw
        self.start_date = start_date
        self.end_date = end_date
        self.max_pages = max_pages

    async def crawl(self):
        data = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()

            await page.goto('https://nid.naver.com/nidlogin.login')
            await page.fill('input[name="id"]', self.naver_id)
            await page.fill('input[name="pw"]', self.naver_pw)
            await page.click('button#log\\.login')
            await page.wait_for_timeout(2000)

            await page.goto(self.cafe_url)

            for key in tqdm(self.keywords, desc="네이버 카페 크롤링"):
                await page.locator('#topLayerQueryInput').fill(key)
                await page.keyboard.press('Enter')
                await page.wait_for_timeout(2000)

                iframe = page.frame(name="cafe_main")
                if not iframe: continue

                # 날짜 설정
                await iframe.locator('#currentSearchDateTop').click()
                await iframe.locator('#input_1_top').fill(self.start_date)
                await iframe.locator('#input_2_top').fill(self.end_date)
                await iframe.locator('#btn_set_top').click()
                await iframe.locator('button.btn-search-green').click()
                await page.wait_for_timeout(2000)

                for _ in range(self.max_pages):
                    post_links = await iframe.locator('a.article').all()
                    urls_to_visit = [await link.get_attribute('href') for link in post_links]
                    
                    for link_url in urls_to_visit:
                        new_page = await context.new_page()
                        await new_page.goto(f"https://cafe.naver.com{link_url}")
                        await new_page.wait_for_timeout(1000)
                        
                        try:
                            frame = new_page.frame(name="cafe_main")
                            if frame:
                                soup = bs(await frame.content(), "lxml")
                                title = soup.select_one("h3.title_text").text.strip()
                                content_area = soup.select_one("div.se-main-container, div.ContentRenderer")
                                content = content_area.get_text(strip=True) if content_area else ''
                                date = soup.select_one("span.date").text.strip()
                                comments_tags = soup.select("span.text_comment")
                                comments = " | ".join([c.text.strip() for c in comments_tags])
                                
                                data.append({
                                    "date": date, "keyword": key, "title": title, "contents": content,
                                    "comments": comments, "site": "네이버카페", "url": new_page.url
                                })
                        except Exception as e:
                            print(f"카페 게시글 파싱 오류: {e}")
                        finally:
                            await new_page.close()
                    
                    try:
                      next_button = iframe.locator('a:text("다음")')
                      if await next_button.is_visible():
                          await next_button.click()
                          await page.wait_for_timeout(2000)
                      else:
                          break
                    except:
                      break

            await browser.close()
        return self._save_to_df(data, ["date", "keyword", "title", "contents", "comments", "site", "url"])


class BoardCrawler(CafeCrawler):
    """특정 카페의 지정된 게시판에서 키워드로 게시글을 크롤링합니다."""
    def __init__(self, keywords, board_url_format, naver_id, naver_pw):
        super().__init__(keywords, '', naver_id, naver_pw, '', '', 2) # Base class init
        self.board_url_format = board_url_format
        
    async def crawl(self):
        data = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()

            await page.goto('https://nid.naver.com/nidlogin.login')
            await page.fill('input[name="id"]', self.naver_id)
            await page.fill('input[name="pw"]', self.naver_pw)
            await page.click('button#log\\.login')
            await page.wait_for_timeout(2000)

            for key in tqdm(self.keywords, desc="지정 게시판 크롤링"):
                for page_num in range(1, self.max_pages + 1):
                    url = self.board_url_format.format(keyword=key, page=page_num)
                    await page.goto(url)
                    await page.wait_for_timeout(1000)

                    iframe = page.frame(name="cafe_main")
                    if not iframe: continue
                        
                    post_links = await iframe.locator('a.article').all()
                    urls_to_visit = [f"https://cafe.naver.com{await link.get_attribute('href')}" for link in post_links]
                    if not urls_to_visit: break
                    
                    for link_url in urls_to_visit:
                        new_page = await context.new_page()
                        await new_page.goto(link_url)
                        await new_page.wait_for_timeout(1000)
                        
                        try:
                            frame = new_page.frame(name="cafe_main")
                            if frame:
                                soup = bs(await frame.content(), "lxml")
                                title = soup.select_one("h3.title_text").text.strip()
                                content_area = soup.select_one("div.se-main-container, div.ContentRenderer")
                                content = content_area.get_text(strip=True) if content_area else ''
                                date = soup.select_one("span.date").text.strip()
                                board_title = soup.select_one("div.ArticleTitle a.board_name").text.strip()
                                comments_tags = soup.select("span.text_comment")
                                comments = " | ".join([c.text.strip() for c in comments_tags])
                                
                                data.append({
                                    "date": date, "keyword": key, "title": title, "contents": content,
                                    "comments": comments, "site": "게시판지정", "url": new_page.url,
                                    "board_titles": board_title
                                })
                        except Exception as e:
                            print(f"게시판 게시글 파싱 오류: {e}")
                        finally:
                            await new_page.close()
            
            await browser.close()
        return self._save_to_df(data, ["date", "keyword", "title", "contents", "comments", "site", "url", "board_titles"])