from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from driver_manager import setup_driver
import json

def perform_web_research(query):
    """Perform web research using Selenium and BeautifulSoup."""
    driver = setup_driver()
    try:
        url = f"https://www.google.com/search?q={query}"
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
        )
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        links = [a['href'] for a in soup.select('div.g a') if a['href'].startswith('http')]
        results = []
        for link in links[:3]:  # Limit to first 3 links for brevity
            driver.get(link)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            page_soup = BeautifulSoup(driver.page_source, 'html.parser')
            page_text = ' '.join([p.get_text() for p in page_soup.find_all('p')])
            results.append(page_text[:500])  # Limit to first 500 characters
        return json.dumps({"query": query, "results": results})
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        driver.quit()
