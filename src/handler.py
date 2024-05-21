""" Example handler file. """

import logging
import os
import sys
import time

import runpod
import torch
from dotenv import load_dotenv

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory)

logging.basicConfig(level=logging.INFO)


# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
from language_id_utils import model as language_id_model
from language_id_utils import tokenizer as language_id_tokenizer
from transcribe_utils import (
    get_audio_file,
    setup_decoder,
    setup_model,
    setup_pipeline,
    transcribe_audio,
)
from translate_utils import translate

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transcribe_main(target_lang, audio_file):
    model_id = "Sunbird/sunbird-mms"
    language = target_lang

    try:
        model, tokenizer, processor, feature_extractor = setup_model(model_id, language)
        decoder = setup_decoder(language, tokenizer, feature_extractor)
        pipe = setup_pipeline(
            model, language, tokenizer, feature_extractor, processor, decoder
        )
        transcription = transcribe_audio(pipe, audio_file)
        return transcription
    except Exception as e:
        logging.error(f"Error downloading language model file: {e}")
        return None


def translate_task(job_input):
    source_language = job_input.get("source_language")
    target_language = job_input.get("target_language")
    text_to_translate = job_input.get("text")

    if not (source_language and target_language and text_to_translate):
        raise ValueError("Missing required translation parameters")

    translated_text = translate(text_to_translate, source_language, target_language)
    return {"text": text_to_translate, "translated_text": translated_text}


def transcribe_task(job_input):
    target_lang = job_input.get("target_lang", "lug")
    audio_file_path = job_input.get("audio_file")

    if not audio_file_path:
        raise ValueError("Missing audio file for transcription")

    audio_file = get_audio_file(audio_file_path)

    start_time = time.time()
    transcription = transcribe_main(target_lang, audio_file)
    end_time = time.time()
    execution_time = end_time - start_time

    logging.info(
        f"Audio transcription execution time: {execution_time:.4f} seconds / {execution_time / 60:.4f} minutes"
    )

    return {"audio_transcription": transcription.get("text")}


def auto_detect_language_task(job_input):
    text = job_input.get("text")

    if not text:
        raise ValueError("Missing text for language detection")

    # Note that we convert the text input to lower case
    inputs = language_id_tokenizer(text.lower(), return_tensors="pt").to(device)
    output = language_id_model.to(device).generate(**inputs, max_new_tokens=5)
    result = language_id_tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return {"language": result}


def handler(job):
    """Handler function that processes jobs."""
    job_input = job.get("input")
    if not job_input:
        return {"Error": "Job input is missing"}

    task = job_input.get("task")
    if not task:
        return {"Error": "Task is missing from job input"}

    try:
        if task == "translate":
            return translate_task(job_input)
        elif task == "transcribe":
            return transcribe_task(job_input)
        elif task == "auto_detect_language":
            return auto_detect_language_task(job_input)
        else:
            return {"Error": f"Unknown task: {task}"}
    except Exception as e:
        logging.error(f"Error processing task {task}: {e}")
        return {"Error": str(e)}


runpod.serverless.start({"handler": handler})
