from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

service = Service(executable_path='chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_argument('--headless')

driver = webdriver.Chrome(service=service, options=options)

for year in range(2023, 2025):
    for month in range(1, 13):
        links = dict()
        driver.get(f"https://economictimes.indiatimes.com/archive/year-{year},month-{month}.cms")
        table = driver.find_element(By.TAG_NAME, "table")
        anchors = table.find_elements(By.TAG_NAME, 'a')
        for anchor in anchors:
            date = f"{anchor.text if anchor.text != '' else '1'}-{month}-{year}"
            links[date] = anchor.get_attribute('href')
        news = []
        for (date, news_link) in links.items():
            try:
                driver.get(news_link)
                time.sleep(1)
                page = driver.find_element(By.ID,'pageContent')
                table = page.find_element(By.CLASS_NAME, 'content')
                anchors = table.find_elements(By.TAG_NAME, 'a')
                current_news = [element.text for element in anchors]
                current_news.insert(0, date+'\n')
                current_news.append('\n')
                news.extend(current_news)
            except:
                print("The following date is not recorded", date)

        date = str(month)+'-'+str(year)
        print(date)
        with open('./news/'+date+'.txt', 'w+') as file:
                file.write(' '.join(news))
