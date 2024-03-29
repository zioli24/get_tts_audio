"""
codes from https://github.com/adelacvg/ttts
"""
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

service = Service(executable_path=ChromeDriverManager().install())
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=chrome_options,service=service)


websites = [
    "https://player.fm/series/cbc-news-hourly-edition/the-world-this-hour-for-20240307-at-2200-est",
    "https://zh.player.fm/series/fm-59854",
]


urls_file = "./urls.txt"
for website in websites:
    driver.get(website)

    scrolls = 10
    for _ in range(scrolls):
        body = driver.find_element(By.TAG_NAME,'html')
        body.send_keys(Keys.END)
        time.sleep(2)
        body.send_keys(Keys.PAGE_UP)  # 模拟按下“Page Up”键，将页面稍微向上滑动
        time.sleep(1)  # 等待一段时间，确保页面加载完成

    html_content = driver.page_source

    soup = BeautifulSoup(html_content, "html.parser")

    target_tags = soup.select('a[href$=".m4a"]')
    # audio_links = soup.find_all("audio")
    i = 0
    for tag in target_tags:
        print(tag)
        i = 1-i
        if i==0:
            continue
        audio_url = tag['href']
        with open(urls_file, "a+") as file:
            file.write(audio_url + "\n")

driver.quit()