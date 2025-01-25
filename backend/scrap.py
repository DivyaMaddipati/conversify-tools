from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv
import time

# Setup WebDriver
driver = webdriver.Chrome()
driver.maximize_window()  # Maximize the browser window for better interaction

# URL to scrape
url = "https://svecw.edu.in/placement-details/"
output_file = "placement_details_complete.csv"

# Categories and their corresponding Full XPaths
categories = {
    "2020-2024": None,  # Default open section
}

# Function to extract table data
def extract_table_data(academic_year):
    table_data = []
    try:
        # Locate the Show Entries dropdown and select 'All'
        show_entries_dropdown = Select(WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "select"))
        ))
        show_entries_dropdown.select_by_visible_text("All")
        time.sleep(3)  # Wait for table to reload with all data
        
        # Locate the table
        table = driver.find_element(By.TAG_NAME, "table")
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        # Extract table rows (skip header row)
        for row in rows[1:]:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 3:  # Ensure valid row
                company_name = cells[1].text.strip()
                no_of_selects = cells[2].text.strip()
                salary_lpa = cells[3].text.strip()
                table_data.append([academic_year, company_name, no_of_selects, salary_lpa])
    except Exception as e:
        print(f"Error extracting data for {academic_year}: {e}")
    return table_data

try:
    # Open the URL
    driver.get(url)
    time.sleep(5)  # Allow the page to load completely
    
    all_data = [["Academic Year", "Company Name", "No of Selects", "Salary(LPA)"]]  # CSV Headers
    
    # Process each category
    for academic_year, xpath in categories.items():
        print(f"Processing data for: {academic_year}")
        
        # Expand the section if not already open
        if xpath:
            plus_icon = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            driver.execute_script("arguments[0].click();", plus_icon)  # Use JS click
            time.sleep(2)  # Allow the section to expand
        
        # Extract table data for the current category
        data = extract_table_data(academic_year)
        all_data.extend(data)
    
    # Write all collected data to CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(all_data)
    
    print(f"Data successfully saved to '{output_file}'.")

except Exception as e:
    print("An error occurred:", e)

finally:
    driver.quit()  # Close the browser
