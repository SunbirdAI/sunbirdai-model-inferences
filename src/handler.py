""" Example handler file. """

import logging
import math
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
from asr_summarization_utils import process_and_correct_text
from asr_summarization_utils import translate as asr_summarise
from diarization_utils import format_diarization_output, process_audio_diarization
from language_id_utils import model as language_id_model
from language_id_utils import predict as classify_predict
from language_id_utils import tokenizer as language_id_tokenizer
from summarization_utils import summarize_text
from transcribe_utils import (
    get_audio_file,
    setup_decoder,
    setup_model,
    setup_pipeline,
    transcribe_audio,
)
from transcribe_whisper_utils import (
    WhisperASR,
    get_audio_file_info,
    remove_audio_silence,
)
from translate_utils import process_and_translate_text

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


def transcribe_whisper(target_lang, audio_file):
    model_id = "jq/whisper-large-v2-multilingual"
    whisper = WhisperASR(model_id)
    processor, model = whisper.setup_model()
    language_code = whisper.get_language_code(target_lang, processor)

    if os.path.exists(audio_file):
        file_size_mb, duration_minutes = get_audio_file_info(audio_file)
        logging.info(
            f"File size: {file_size_mb}mb Duration in minutes: {duration_minutes}"
        )
        if math.floor(duration_minutes) > 1:
            pipeline = whisper.setup_pipeline(
                model, processor, language_code=language_code, batch=True
            )
        else:
            pipeline = whisper.setup_pipeline(
                model, processor, language_code=language_code, batch=False
            )

        try:
            transcription = whisper.transcribe_audio(
                audio_file, pipeline, return_timestamps=True
            )
            return transcription
        except Exception as e:
            logging.error(str(e))
            return None
    else:
        logging.error(f"Error downloading language model file: {e}")
        return None


def transcribe_whisper_large_v2_multilingual_prompts_corrected(audio_file):
    whisper = WhisperASR(
        model_path="jq/whisper-large-v2-multilingual-prompts-corrected"
    )
    pipeline = whisper.setup_pipeline_whisper_large_v2_multilingual_prompts_corrected()
    generate_kwargs = whisper.generate_transcribe_kwargs(pipeline, device)
    audio_file = remove_audio_silence(audio_file_path=audio_file)
    transcription = pipeline(
        audio_file, generate_kwargs=generate_kwargs, return_timestamps=True
    )
    return transcription


def translate_task(job_input):
    source_language = job_input.get("source_language")
    target_language = job_input.get("target_language")
    text_to_translate = job_input.get("text")

    if not (source_language and target_language and text_to_translate):
        raise ValueError("Missing required translation parameters")

    translated_text = process_and_translate_text(
        text_to_translate, source_language, target_language
    )
    return {"text": text_to_translate, "translated_text": translated_text}


def transcribe_task(job_input):
    response = {}
    target_lang = job_input.get("target_lang", "lug")
    audio_file_path = job_input.get("audio_file")
    recognise_speakers = job_input.get("recognise_speakers", False)
    use_whisper = job_input.get("whisper", False)
    organisation = job_input.get("organisation", False)

    if not audio_file_path:
        raise ValueError("Missing audio file for transcription")

    audio_file = get_audio_file(audio_file_path)

    start_time = time.time()
    if use_whisper:
        transcription = transcribe_whisper(target_lang, audio_file)
        transcription_text = transcription.get("text")
    elif organisation:
        transcription = transcribe_whisper_large_v2_multilingual_prompts_corrected(
            audio_file
        )
        transcription_text = transcription.get("text")
    else:
        transcription = transcribe_main(target_lang, audio_file)
        transcription_text = transcription.get("text")

        if target_lang in ["eng", "lug"]:
            transcription_text = process_and_correct_text(
                transcription_text, chunk_size=50, source_language=target_lang
            )

    end_time = time.time()
    execution_time = end_time - start_time

    response["audio_transcription"] = transcription_text

    if recognise_speakers:
        hf_token = os.getenv("HF_TOKEN")
        diarization_output = process_audio_diarization(
            audio_file, hf_token, transcription, device
        )
        formatted_diarization_output = format_diarization_output(diarization_output)
        response["diarization_output"] = diarization_output
        response["formatted_diarization_output"] = formatted_diarization_output

    logging.info(
        f"Audio transcription execution time: {execution_time:.4f} seconds / {execution_time / 60:.4f} minutes"
    )

    return response


def asr_summarise_task(job_input):
    source_language = job_input.get("source_language")
    target_language = job_input.get("target_language")
    text = job_input.get("text")

    if not (source_language and target_language and text):
        raise ValueError("Missing required translation parameters")

    corrected_text = process_and_correct_text(
        text, chunk_size=50, source_language=source_language
    )
    summary = asr_summarise("<summary> " + text, source_language, target_language)
    short_summary = asr_summarise(
        "<shortsummary> " + text, source_language, target_language
    )
    return {
        "corrected_text": corrected_text,
        "summary": summary,
        "short_summary": short_summary,
    }


def auto_detect_language_task(job_input):
    text = job_input.get("text")

    if not text:
        raise ValueError("Missing text for language detection")

    # Note that we convert the text input to lower case
    inputs = language_id_tokenizer(text.lower(), return_tensors="pt").to(device)
    output = language_id_model.to(device).generate(**inputs, max_new_tokens=5)
    result = language_id_tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return {"language": result}


def language_classification_task(job_input):
    text = job_input.get("text")

    if not text:
        raise ValueError("Missing text for language classification")

    result = classify_predict(text, device)

    return {"predictions": result}


def summarization_task(job_input):
    text = job_input.get("text")

    if not text:
        raise ValueError("Missing text for summarization")

    summary = summarize_text(text)

    return {"summarized_text": summary}


def handler(job):
    """Handler function that processes jobs."""
    job_input = job.get("input")
    if not job_input:
        return {"Error": "Job input is missing"}

    task = job_input.get("task")
    if not task:
        return {"Error": "Task is missing from job input"}

    task_map = {
        "translate": translate_task,
        "transcribe": transcribe_task,
        "asr_summarise": asr_summarise_task,
        "auto_detect_language": auto_detect_language_task,
        "language_classify": language_classification_task,
        "summarise": summarization_task,
    }

    if task not in task_map:
        return {"Error": f"Unknown task: {task}"}

    try:
        return task_map[task](job_input)
    except Exception as e:
        logging.error(f"Error processing task {task}: {e}")
        return {"Error": str(e)}


runpod.serverless.start({"handler": handler})
