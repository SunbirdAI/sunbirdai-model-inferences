import logging
import os

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging

sess = requests.session()

retries = Retry(
    total=5, backoff_factor=1, status_forcelist=[400, 429, 500, 502, 503, 504]
)

sess.mount("https://", HTTPAdapter(max_retries=retries))


class AITasks:
    """
    A class for performing AI tasks such as transcription, translation and anonymised
    summarisation using a REST API.

    Parameters:
        base_url (str): The base URL of the API.
        auth_token (str): The authentication token for accessing the API.
    """

    def __init__(self, base_url, auth_token):
        self.base_url = base_url
        self.auth_token = auth_token

    def transcribe(self, endpoint, audio_path, language):
        """
        Transcribes audio file to text.

        Parameters:
            endpoint (str): The endpoint to be concatenated to the base URL.
            audio_path (str): The path to the audio file.
            language (str): The language of the audio.

        Returns:
            str: The transcription text.
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }
        file_name = os.path.basename(audio_path)
        files = {
            "audio": (file_name, open(audio_path, "rb"), "audio/mpeg"),
        }
        data = {
            "language": language,
            "adapter": language,
            "whisper": True,
        }

        try:
            response = sess.post(url, headers=headers, files=files, data=data)
            # print(response.content)
            return response.json()["audio_transcription"]
        except Exception as e:
            # For big transcriptions
            logger.error(f"{str(e)}: Transcription skipped")
            return None

    def translate(self, endpoint, source_language, target_language, text):
        """
        Translates text from source language to target language.

        Parameters:
            endpoint (str): The endpoint to be concatenated to the base URL.
            source_language (str): The language of the text to be translated.
            target_language (str): The language to translate the text to.
            text (str): The text to be translated.

        Returns:
            str: The translated text.
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }
        data = {
            "source_language": source_language,
            "target_language": target_language,
            "text": text,
        }

        response = sess.post(url, headers=headers, json=data)
        return response.json()["output"]["translated_text"]

    def summarise(self, endpoint, text):
        """
        Anonymously summarises the given text. The supported languages are
        English and Luganda for now.

        Parameters:
            endpoint (str): The endpoint to be concatenated to the base URL.
            text (str): The text to be summarised.

        Returns:
            str: The summarised text.
        """
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }
        data = {"text": text}

        response = sess.post(url, headers=headers, json=data)
        return response.json()["summarized_text"]


# Example Usage
if __name__ == "__main__":
    # Initialize AITasks object
    token = os.getenv("AUTH_TOKEN")
    ai_tasks = AITasks(
        base_url="https://api.sunbird.ai/tasks",
        auth_token=token,
    )

    # Transcribe audio
    transcription = ai_tasks.transcribe(
        endpoint="stt", audio_path="./trac_fm/ach/MEGA 12.1.mp3", language="ach"
    )
    print(f"Transcription: {transcription}")

    # Translate text
    text = """
        nyia florence leonat ga aul tok toa laik maidota aweno i latin ginie ni i torido apin 
        nao e ni i ic fomi wek ando atekiŋ ova latinneni in yugoan fenicdic ginene ni ci dok cito 
        kwano aloko kwede mamol pien kit n aloko ki minne kwede wariwane minne paka nio en ditti en 
        aye i cecemuel mumito acak fidiŋ en bene core doŋ ocito yaco i coke ana cud faid pode wer ni 
        timoo wodok eskul pien ana i kanoticer ni a witimo gini kuman in doŋ tuel wid yu wot anebritiŋ 
        ci noŋo e onipati ka pwony noŋo atye ka neko gaŋ aye
    """
    source_language = "ach"
    target_language = "eng"
    translation = ai_tasks.translate(
        endpoint="nllb_translate",
        source_language=source_language,
        target_language=target_language,
        text=text,
    )
    print(f"Translation: {translation}")

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

    summary = ai_tasks.summarise(
        endpoint="summarise",
        text=text,
    )
    print(f"Summarisation: {summary}")
