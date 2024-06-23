import logging
import os

import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)


def read_text_from_file(file_path):
    """
    Reads text from a .txt file and returns the content as a string.

    Args:
        file_path (str): The path to the .txt file.

    Returns:
        str: The content of the file as a string.

    Example:
        >>> file_path = 'example.txt'
        >>> content = read_text_from_file(file_path)
        >>> print(content)
        This is an example text file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found. Please check the file path and try again."
    except Exception as e:
        return f"An error occurred: {e}"


def write_text_to_file(text: str, file_path: str) -> None:
    """
    Writes the provided text to a .txt file.

    Args:
        text (str): The text to be written to the file.
        file_path (str): The path where the .txt file will be created. This includes the file name and extension.

    Returns:
        None

    Example:
        text = "This is an example text to be written to a file."
        file_path = "example.txt"
        write_text_to_file(text, file_path)
    """
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)


def translate(text, source_language, target_language):
    url = "https://api.sunbird.ai/tasks/nllb_translate"
    token = os.getenv("AUTH_TOKEN")
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    data = {
        "source_language": source_language,
        "target_language": target_language,
        "text": text,
    }

    response = requests.post(url, headers=headers, json=data)

    return response.json()["output"].get("translated_text")


if __name__ == "__main__":
    # Example usage
    source_language = "ach"
    target_language = "eng"
    text = (
        "nyia florence leonat ga aul tok toa laik maidota aweno i "
        "latin ginie ni i torido apin nao e ni i ic fomi wek ando "
        "atekiŋ ova latinneni in yugoan fenicdic ginene ni ci dok "
        "cito kwano aloko kwede mamol pien kit n aloko ki minne kwede "
        "wariwane minne paka nio en ditti en aye i cecemuel mumito acak "
        "fidiŋ en bene core doŋ ocito yaco i coke ana cud faid pode wer "
        "ni timoo wodok eskul pien ana i kanoticer ni a witimo gini kuman "
        "in doŋ tuel wid yu wot anebritiŋ ci noŋo e onipati ka pwony noŋo "
        "atye ka neko gaŋ aye"
    )
    translated_text = translate(text, source_language, target_language)
    print(translated_text)
    print("=" * 100)

    file_path = "transcript.txt"
    content = read_text_from_file(file_path)

    source_language = "eng"
    target_language = "ach"
    file_output = f"transcipt_{source_language}_{target_language}.txt"
    translated_text = translate(content, source_language, target_language)

    write_text_to_file(translated_text, file_output)
    logging.info(f"Translated to file {file_output}")
    print("=" * 100)

    source_language = "eng"
    target_language = "lug"
    file_output = f"transcipt_{source_language}_{target_language}.txt"
    translated_text = translate(content, source_language, target_language)

    write_text_to_file(translated_text, file_output)
    logging.info(f"Translated to file {file_output}")
