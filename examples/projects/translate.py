import os

import requests
from dotenv import load_dotenv

load_dotenv()

url = "https://api.sunbird.ai/tasks/nllb_translate"
token = os.getenv("AUTH_TOKEN")
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

data = {
    "task": "translate",
    "source_language": "ach",
    "target_language": "eng",
    "text": "nyia florence leonat ga aul tok toa laik maidota aweno i latin ginie ni i torido apin nao e ni i ic fomi wek ando atekiŋ ova latinneni in yugoan fenicdic ginene ni ci dok cito kwano aloko kwede mamol pien kit n aloko ki minne kwede wariwane minne paka nio en ditti en aye i cecemuel mumito acak fidiŋ en bene core doŋ ocito yaco i coke ana cud faid pode wer ni timoo wodok eskul pien ana i kanoticer ni a witimo gini kuman in doŋ tuel wid yu wot anebritiŋ ci noŋo e onipati ka pwony noŋo atye ka neko gaŋ aye",
}

response = requests.post(url, headers=headers, json=data)

print(response.json())
