import logging
import math
import os
import time
from typing import Tuple

import requests
import runpod
from dotenv import load_dotenv
from google.cloud import storage
from pydub import AudioSegment

load_dotenv()
logging.basicConfig(level=logging.INFO)
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
# Set RunPod API Key
runpod.api_key = os.getenv("RUNPOD_API_KEY")


def upload_audio_file(file_path):
    try:
        # Initialize a client and get the bucket
        storage_client = storage.Client()
        bucket_name = os.getenv("AUDIO_CONTENT_BUCKET_NAME")
        bucket = storage_client.bucket(bucket_name)

        blob_name = os.path.basename(file_path)

        # Upload the file to the bucket
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)

        return blob_name
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_audio_file_info(file_path: str) -> Tuple[float, float]:
    """
    Returns the size of the audio file in MBs and the duration of the audio file in minutes.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        tuple: A tuple containing the size of the file in MBs and the duration of the audio file in minutes.

    Example:
        file_size_mb, duration_minutes = get_audio_file_info("path/to/audio/file.mp3")
        print(f"File Size: {file_size_mb} MB, Duration: {duration_minutes} minutes")
    """
    # Get the file size in MBs
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Load the audio file and get its duration in minutes
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000  # Convert milliseconds to seconds
    duration_minutes = duration_seconds / 60  # Convert seconds to minutes

    return file_size_mb, duration_minutes


def transcription_runpod(audio_file_path: str, language: str, adapter: str) -> str:
    """
    Transcribes audio using the Runpod endpoint.

    This function sends an audio file to the Runpod endpoint for transcription. It specifies the task as "transcribe",
    along with the target language and adapter. It handles potential timeouts and logs the response and elapsed time.

    Args:
        audio_file_path (str): The path to the audio file to be transcribed.
        language (str): The target language for transcription.
        adapter (str): The adapter to be used for transcription.

    Returns:
        str: The transcribed text from the audio file.

    Raises:
        TimeoutError: If the request to the endpoint times out.

    Example:
        transcription = transcription_runpod("path/to/audio/file.mp3", "en", "adapter_name")
        print(transcription)
    """
    endpoint = runpod.Endpoint(RUNPOD_ENDPOINT_ID)

    blob_name = upload_audio_file(file_path=audio_file_path)
    audio_file = blob_name
    logging.info(f"Filename: {audio_file}")
    request_response = {}

    start_time = time.time()
    try:
        request_response = endpoint.run_sync(
            {
                "input": {
                    "task": "transcribe",
                    "target_lang": language,
                    "adapter": adapter,
                    "audio_file": audio_file,
                }
            },
            timeout=900,  # Timeout in seconds.
        )
    except TimeoutError:
        logging.error("Job timed out.")

    end_time = time.time()
    # logging.info(f"Response: {request_response}")

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time} seconds")
    audio_transcription = request_response.get("audio_transcription")
    return audio_transcription


def transcription_api_endpoint(
    audio_file_path: str, language: str, adapter: str
) -> str:
    """
    Transcribes audio using the Sunbird AI API endpoint.

    This function sends an audio file to the Sunbird AI API endpoint for transcription. It includes the file,
    target language, and adapter in the request. It retrieves the transcribed text from the JSON response.

    Args:
        audio_file_path (str): The path to the audio file to be transcribed.
        language (str): The target language for transcription.
        adapter (str): The adapter to be used for transcription.

    Returns:
        str: The transcribed text from the audio file.

    Raises:
        Exception: If there is an error with the request or response.

    Example:
        transcription = transcription_api_endpoint("path/to/audio/file.mp3", "en", "adapter_name")
        print(transcription)
    """
    file_name = os.path.basename(audio_file_path)
    logging.info(f"Filename: {file_name}")
    url = "https://api.sunbird.ai/tasks/stt"
    token = os.getenv("AUTH_TOKEN")
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    files = {
        "audio": (
            file_name,
            open(audio_file_path, "rb"),
            "audio/mpeg",
        ),
    }
    data = {
        "language": language,
        "adapter": adapter,
    }

    response = requests.post(url, headers=headers, files=files, data=data)
    audio_transcription = response.json().get("audio_transcription")
    return audio_transcription


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


def transcribe(audio_file_path: str, language: str, adapter: str) -> None:
    """
    Transcribes an audio file using either the Runpod endpoint or the Sunbird AI API endpoint based on audio duration.

    This function determines the size and duration of the audio file. If the duration exceeds 10 minutes, it uses the
    Runpod endpoint for transcription; otherwise, it uses the Sunbird AI API endpoint. The transcribed text is then
    written to a file named "transcript.txt".

    Args:
        audio_file_path (str): The path to the audio file to be transcribed.
        language (str): The target language for transcription.
        adapter (str): The adapter to be used for transcription.

    Returns:
        None

    Example:
        transcribe("path/to/audio/file.mp3", "eng", "adapter_name")
    """
    filename = os.path.basename(audio_file_path)
    file_size_mb, duration_minutes = get_audio_file_info(audio_file_path)
    logging.info(f"File size: {file_size_mb}mb Duration in minutes: {duration_minutes}")

    if math.ceil(duration_minutes) > 10:
        logging.info("Transcribing from runpod endpoint")
        transcription = transcription_runpod(audio_file_path, language, adapter)
    else:
        transcription = transcription_api_endpoint(audio_file_path, language, adapter)

    transcription_file_output = "transcript.txt"
    write_text_to_file(transcription, transcription_file_output)
    logging.info(f"Transcribed {filename} to {transcription_file_output}")


if __name__ == "__main__":
    audio_file_path = "../../content/Lutino weng twero pwonye episode 32 final.mp3"
    transcribe(audio_file_path, "ach", "ach")
