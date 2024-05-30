import os

import requests
from dotenv import load_dotenv

load_dotenv()

url = "https://api.sunbird.ai/tasks/language_id"
token = os.getenv("AUTH_TOKEN")
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

text = "ndowooza yange ku baana bano abato abatalina tufuna funa ya uganda butuufu"

data = {"text": text}

response = requests.post(url, headers=headers, json=data)

print(response.json())
