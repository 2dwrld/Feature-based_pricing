import requests
from bs4 import BeautifulSoup
import re
import time

# Replace with your actual 2Captcha API key
RUCAPTCHA_API_KEY = 'cd92fce3287cd6a04d911e624b67a96b'
YANDEX_SITE_KEY = 'FEXfAbHQsToo97VidNVk3j4dC74nGW1DgdxjtNB9'


def solve_captcha(api_key, site_key, page_url):
    print("Submitting CAPTCHA to RuCaptcha...")
    response = requests.post(
        'http://rucaptcha.com/in.php',
        data={
            'key': api_key,
            'method': 'yandex',
            'sitekey': site_key,
            'pageurl': page_url,
            'json': 1
        }
    )

    if response.status_code != 200:
        raise Exception(f"Error submitting CAPTCHA to RuCaptcha: {response.status_code}")

    response_data = response.json()
    if response_data.get('status') != 1:
        raise Exception(f"Error submitting CAPTCHA to RuCaptcha: {response_data.get('request')}")

    captcha_id = response_data.get('request')
    print(f"CAPTCHA submitted. ID: {captcha_id}")

    # Get the answer token from 2Captcha
    while True:
        print("Waiting for CAPTCHA solution...")
        response = requests.get(
            'http://rucaptcha.com/res.php',
            params={
                'key': api_key,
                'action': 'get',
                'id': captcha_id,
                'json': 1
            }
        )

        response_data = response.json()
        if response_data.get('status') == 1:
            break
        elif response_data.get('request') != 'CAPCHA_NOT_READY':
            raise Exception(f"Error fetching CAPTCHA solution from RuCaptcha: {response_data.get('request')}")

        time.sleep(5)  # Wait for 5 seconds before checking again

    token = response_data.get('request')
    print(f"CAPTCHA solved. Token: {token}")
    return token


def get_car_listings(make=None, body_style=None, drive_type=None, engine_size=None, horsepower=None):
    # Construct the URL with the provided parameters
    base_url = "https://auto.ru/cars/"

    make_mapping = {
        'alfa-romero': 'alfa_romeo', 'audi': 'audi', 'bmw': 'bmw', 'chevrolet': 'chevrolet', 'dodge': 'dodge',
        'honda': 'honda', 'isuzu': 'isuzu', 'jaguar': 'jaguar', 'mazda': 'mazda', 'mercedes-benz': 'mercedes',
        'mercury': 'mercury', 'mitsubishi': 'mitsubishi', 'nissan': 'nissan', 'peugot': 'peugeot',
        'plymouth': 'plymouth',
        'porsche': 'porsche', 'renault': 'renault', 'saab': 'saab', 'subaru': 'subaru', 'toyota': 'toyota',
        'volkswagen': 'volkswagen', 'volvo': 'volvo'
    }

    if make and make.lower() in make_mapping:
        base_url += f"{make_mapping[make.lower()]}/all/"
    else:
        base_url += "all/"

    body_style_mapping = {
        'convertible': 'body-cabrio/', 'hardtop': '', 'hatchback': '', 'sedan': 'body-sedan/', 'wagon': 'body-wagon/'
    }

    if body_style and body_style.lower() in body_style_mapping:
        base_url += body_style_mapping[body_style.lower()]

    drive_type_mapping = {
        '4wd': '?gear_type=ALL_WHEEL_DRIVE', 'fwd': '?gear_type=FORWARD_CONTROL', 'rwd': '?gear_type=REAR_DRIVE'
    }

    if drive_type and drive_type.lower() in drive_type_mapping:
        base_url += drive_type_mapping[drive_type.lower()]
    else:
        base_url += '?'

    if engine_size:
        base_url += f"&displacement_from={int(engine_size * 10)}"

    if horsepower:
        base_url += f"&power_from={int(horsepower)}"

    print(f"Constructed URL: {base_url}")

    try:
        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        if "SmartCaptcha" in soup.text:
            print("CAPTCHA detected. Trying to solve it...")
            try:
                captcha_solution = solve_captcha(RUCAPTCHA_API_KEY, YANDEX_SITE_KEY, base_url)
            except Exception as e:
                print(f"Error solving CAPTCHA: {e}")
                return []

            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'g-recaptcha-response': captcha_solution
            }
            response = requests.get(base_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

        cars = []
        count = 0
        for listing in soup.find_all('div', class_='ListingItem'):
            print("Processing a car listing...")  # Debug print
            price_block = listing.find('div', class_='ListingItem__priceBlock')
            if price_block:
                print("Found price block...")  # Debug print
                price_content = price_block.find('div', class_='ListingItemPrice__content')
                if price_content:
                    print("Found price content...")  # Debug print
                    price_text = price_content.get_text(strip=True)
                    print(f"Price text: {price_text}")  # Debug print
                    try:
                        price = int(re.sub(r'[^\d]', '', price_text))
                        cars.append(price)
                        count += 1
                    except ValueError:
                        print(f"Could not convert price text to int: {price_text}")  # Debug print  # Debug print
        return cars

    except requests.RequestException as e:
        print(f"HTTP request failed: {e}")
        return []

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
