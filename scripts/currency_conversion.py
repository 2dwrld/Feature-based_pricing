import requests

# Функция для получения курса обмена USD к RUB
def get_usd_to_rub_exchange_rate():
    url = "https://open.er-api.com/v6/latest/RUB"
    response = requests.get(url)
    data = response.json()
    return data['rates']['USD']
