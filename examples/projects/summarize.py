import os

import requests
from dotenv import load_dotenv

load_dotenv()

url = "https://api.sunbird.ai/tasks/summarise"
token = os.getenv("AUTH_TOKEN")
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

text = (
    "ndowooza yange ku baana bano abato abatalina tufuna funa ya uganda butuufu "
    "eserbamby omwana oyo bingi bye yeegomba okuva mu buto bwe ate by'atasobola "
    "kwetuusaako bw'afuna mu naawumuwaamagezi nti ekya mazima nze kaboyiaadeyaatei "
    "ebintu kati bisusse mu uganda wano ebyegombebw'omwana by'atasobola kwetuusaako "
    "ng'ate abazadde nabo bambi bwe beetunulamubamufuna mpola tebasobola kulabirira "
    "mwana oyo bintu by'ayagala ekivaamu omwana akemererwan'ayagala omulenzi omulenzi "
    "naye n'atoba okuatejukira ba mbi ba tannategeera bigambo bya kufuna famire fulani "
    "bakola kyagenda layivu n'afuna embuto eky'amazima nze mbadde nsaba be kikwata "
    "govenment sembera embeera etuyisa nnyo abaana ne tubafaako embeera gwe nyiga gwa "
    "omuzadde olina olabirira maama we olina olabirira n'abato kati kano akasuumuseemu "
    "bwe ka kubulako ne keegulirayooba kapalaobakakioba tokyabisobola ne keyiiyabatuyambe "
    "buduufuembeera bagikyusa mu tulemye"
)

data = {"text": text}

response = requests.post(url, headers=headers, json=data)

print(response.json())
