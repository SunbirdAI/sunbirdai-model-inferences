import os

import requests
from dotenv import load_dotenv

load_dotenv()

url = "https://api.sunbird.ai/tasks/stt"
token = os.getenv("AUTH_TOKEN")
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {token}",
}

files = {
    "audio": (
        "MEGA 12.1.mp3",
        open("./languages/ach/MEGA 12.1.mp3", "rb"),
        "audio/mpeg",
    ),
}
data = {
    "language": "lug",
    "adapter": "lug",
}

response = requests.post(url, headers=headers, files=files, data=data)

print(response.json())
