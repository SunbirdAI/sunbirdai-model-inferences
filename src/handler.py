""" Example handler file. """

import os
import sys
import time

import runpod
from dotenv import load_dotenv

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory)

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
from transcribe_utils import (
    get_audio_file,
    setup_decoder,
    setup_model,
    setup_pipeline,
    transcribe_audio,
)
from translate_utils import translate

load_dotenv()


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
        print(f"Error downloading language model file: {e}")
        return None


def handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job["input"]
    task = job_input.get("task")
    response = {}

    if task == "translate":
        source_language = job_input.get("source_language")
        target_language = job_input.get("target_language")
        text_to_translate = job_input.get("text")

        translated_text = translate(text_to_translate, source_language, target_language)

        response = {"text": text_to_translate, "translated_text": translated_text}
    elif task == "transcribe":
        try:
            target_lang = job_input.get("target_lang", "lug")
            audio_file = get_audio_file(job_input.get("audio_file"))

            start_time = time.time()

            transcription = transcribe_main(target_lang, audio_file)
            response = {"audio_transcription": transcription.get("text")}
            end_time = time.time()
            execution_time = end_time - start_time
            print(
                f"Audio transcription execution time: {execution_time:.4f} seconds / {execution_time / 60:.4f} minutes"
            )
        except Exception as e:
            response = {"Error": str(e)}

    return response


runpod.serverless.start({"handler": handler})
