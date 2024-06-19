import requests
from bs4 import BeautifulSoup
import re


def get_car_listings(make, body_style, engine_type, drive_type):
    url = "https://auto.ru/cars/all/"
    params = {
        "mark": make,
        "body_style": body_style,
        "engine_type": engine_type,
        "drive_type": drive_type,
        "price_currency": "RUB"
    }
    response = requests.get(url, params=params)
    soup = BeautifulSoup(response.text, 'html.parser')

    cars = []
    for listing in soup.find_all('div', class_='ListingItem'):
        price_div = listing.find('div', class_='ListingItemPrice__content')
        if price_div:
            price_text = price_div.get_text(strip=True)
            price = int(re.sub(r'[^\d]', '', price_text))  # Удаляем все, кроме цифр
            cars.append(price)

    return cars
