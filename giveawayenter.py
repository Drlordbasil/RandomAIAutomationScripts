# file: giveaway_automation_selenium.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import requests
import time
import os

# User data
user_data = {
    'FirstName': 'John',
    'LastName': 'Doe',
    'Address': '123 Main St',
    'City': 'Anytown',
    'State': 'CA',  # Ensure this matches the dropdown value in the form
    'Zip': '90210',  # Updated field name
    'Phone': '123-456-7890',
    'Email': 'johndoe@example.com',  # Updated field name
    'AgreeCheck': 'true'  # Updated field name for checkbox
}

# URL of the giveaway form
giveaway_url = 'https://www.drpeppercoconutsweeps.com/'

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def log_form_details(driver):
    print("\nForm Details:")
    form_elements = driver.find_elements(By.CSS_SELECTOR, 'form input, form select, form textarea')
    for elem in form_elements:
        elem_name = elem.get_attribute('name')
        elem_type = elem.get_attribute('type')
        elem_value = elem.get_attribute('value')
        elem_class = elem.get_attribute('class')
        elem_id = elem.get_attribute('id')
        print(f"Name: {elem_name}, Type: {elem_type}, Value: {elem_value}, Class: {elem_class}, ID: {elem_id}")

def log_page_source(driver):
    page_source_path = 'page_source.html'
    with open(page_source_path, 'w', encoding='utf-8') as f:
        f.write(driver.page_source)
    print(f"Saved page source as {page_source_path}")

def main():
    # Initialize the Chrome driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    # Open the giveaway page
    driver.get(giveaway_url)
    
    try:
        # Wait for the form to load
        wait = WebDriverWait(driver, 10)
        
        # Fill the form fields
        print("Filling First Name")
        wait.until(EC.presence_of_element_located((By.NAME, 'FirstName'))).send_keys(user_data['FirstName'])
        
        print("Filling Last Name")
        wait.until(EC.presence_of_element_located((By.NAME, 'LastName'))).send_keys(user_data['LastName'])
        
        print("Filling Address")
        wait.until(EC.presence_of_element_located((By.NAME, 'Address'))).send_keys(user_data['Address'])
        
        print("Filling City")
        wait.until(EC.presence_of_element_located((By.NAME, 'City'))).send_keys(user_data['City'])
        
        print("Selecting State")
        state_select = Select(wait.until(EC.presence_of_element_located((By.NAME, 'State'))))
        state_select.select_by_value(user_data['State'])
        
        print("Filling Zip Code")
        wait.until(EC.presence_of_element_located((By.NAME, 'Zip'))).send_keys(user_data['Zip'])
        
        print("Filling Phone")
        wait.until(EC.presence_of_element_located((By.NAME, 'Phone'))).send_keys(user_data['Phone'])
        
        print("Filling Email Address")
        wait.until(EC.presence_of_element_located((By.NAME, 'Email'))).send_keys(user_data['Email'])
        
        print("Clicking 21+ Checkbox")
        agree_checkbox = wait.until(EC.presence_of_element_located((By.NAME, 'AgreeCheck')))
        
        # Ensure checkbox is visible
        driver.execute_script("arguments[0].scrollIntoView(true);", agree_checkbox)
        
        if not agree_checkbox.is_selected():
            driver.execute_script("arguments[0].click();", agree_checkbox)
        
        # Capture CAPTCHA image
        print("Capturing CAPTCHA image")
        captcha_image = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'img[src*="captcha"]')))
        captcha_image_path = 'captcha.png'
        captcha_image.screenshot(captcha_image_path)
        print(f"CAPTCHA image saved as {captcha_image_path}")
        
        # Verify CAPTCHA image
        try:
            with open(captcha_image_path, 'rb') as img:
                print(f"CAPTCHA image file size: {len(img.read())} bytes")
        except Exception as e:
            print(f"Failed to read CAPTCHA image: {e}")
            return
        
        # Solve CAPTCHA
        print("Solving CAPTCHA")
        captcha_text = query(captcha_image_path)
        if captcha_text and isinstance(captcha_text, list) and 'generated_text' in captcha_text[0]:
            captcha_solution = captcha_text[0]['generated_text']
            print(f"CAPTCHA solved: {captcha_solution}")
        else:
            print("Failed to solve CAPTCHA")
            return
        
        # Enter CAPTCHA
        print("Entering CAPTCHA solution")
        captcha_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input[placeholder="Enter text from image"]')))
        captcha_input.send_keys(captcha_solution)
        
        print("Submitting Form")
        wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'form button[type="submit"]'))).click()
        
        # Wait for a few seconds to see the result
        time.sleep(5)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        log_form_details(driver)
        log_page_source(driver)
        driver.save_screenshot('error_screenshot.png')
        print("Saved screenshot of error as error_screenshot.png")
    
    finally:
        # Close the browser
        driver.quit()

if __name__ == "__main__":
    main()
