import datetime
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

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TaskHandler:
    def __init__(self):
        self.device = device

    def transcribe(self, job_input):
        from diarization_utils import (
            format_diarization_output,
            process_audio_diarization,
        )
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
            model_id = "jq/whisper-large-v2-salt-plus-xog-myx-kin-swa-sample-packing"
            whisper = WhisperASR(model_id)
            processor, model = whisper.setup_model()
            language_code = whisper.get_language_code(target_lang, processor)

            file_size_mb, duration_minutes = get_audio_file_info(audio_file)
            if duration_minutes > 1:
                pipeline = whisper.setup_pipeline(
                    model, processor, language_code=language_code, batch=True
                )
            else:
                pipeline = whisper.setup_pipeline(
                    model, processor, language_code=language_code, batch=False
                )

            transcription = whisper.transcribe_audio(
                audio_file, pipeline, return_timestamps=True
            )
            transcription_text = transcription.get("text")
        elif organisation:
            whisper = WhisperASR(
                model_path="jq/whisper-large-v2-multilingual-prompts-corrected"
            )
            pipeline = (
                whisper.setup_pipeline_whisper_large_v2_multilingual_prompts_corrected()
            )
            generate_kwargs = whisper.generate_transcribe_kwargs(pipeline, self.device)
            audio_file = remove_audio_silence(audio_file_path=audio_file)
            transcription = pipeline(
                audio_file, generate_kwargs=generate_kwargs, return_timestamps=True
            )
            transcription_text = transcription.get("text")
        else:
            model_id = "Sunbird/sunbird-mms"
            language = target_lang

            model, tokenizer, processor, feature_extractor = setup_model(
                model_id, language
            )
            decoder = setup_decoder(language, tokenizer, feature_extractor)
            pipe = setup_pipeline(
                model, language, tokenizer, feature_extractor, processor, decoder
            )
            transcription = transcribe_audio(pipe, audio_file)
            transcription_text = transcription.get("text")

        end_time = time.time()
        execution_time = end_time - start_time

        response["audio_transcription"] = transcription_text

        if recognise_speakers:
            hf_token = os.getenv("HF_TOKEN")
            diarization_output = process_audio_diarization(
                audio_file, hf_token, transcription, self.device
            )
            formatted_diarization_output = format_diarization_output(diarization_output)
            response["diarization_output"] = diarization_output
            response["formatted_diarization_output"] = formatted_diarization_output

        logging.info(
            f"Audio transcription execution time: {execution_time:.4f} seconds / {execution_time / 60:.4f} minutes"
        )

        return response

    def translate(self, job_input):
        from translate_utils import process_and_translate_text

        source_language = job_input.get("source_language")
        target_language = job_input.get("target_language")
        text_to_translate = job_input.get("text")

        if not (source_language and target_language and text_to_translate):
            raise ValueError("Missing required translation parameters")

        translated_text = process_and_translate_text(
            text_to_translate, source_language, target_language
        )
        return {"text": text_to_translate, "translated_text": translated_text}

    def asr_summarise(self, job_input):
        from asr_summarization_utils import process_and_correct_text
        from asr_summarization_utils import translate as asr_summarise

        source_language = job_input.get("source_language")
        target_language = job_input.get("target_language")
        text = job_input.get("text")

        if not (source_language and target_language and text):
            raise ValueError("Missing required ASR summarization parameters")

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

    def auto_detect_language(self, job_input):
        from language_id_utils import model as language_id_model
        from language_id_utils import tokenizer as language_id_tokenizer

        text = job_input.get("text")

        if not text:
            raise ValueError("Missing text for language detection")

        inputs = language_id_tokenizer(text.lower(), return_tensors="pt").to(
            self.device
        )
        output = language_id_model.to(self.device).generate(**inputs, max_new_tokens=5)
        result = language_id_tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        return {"language": result}

    def language_classify(self, job_input):
        from language_id_utils import predict as classify_predict

        text = job_input.get("text")

        if not text:
            raise ValueError("Missing text for language classification")

        result = classify_predict(text, self.device)

        return {"predictions": result}

    def summarise(self, job_input):
        from summarization_utils import summarize_text

        text = job_input.get("text")

        if not text:
            raise ValueError("Missing text for summarization")

        summary = summarize_text(text)

        return {"summarized_text": summary}

    def auto_detect_audio_language(self, job_input):
        from transcribe_utils import get_audio_file
        from transcribe_whisper_utils import WhisperASR

        audio_file_path = job_input.get("audio_file")

        if not audio_file_path:
            raise ValueError("Missing audio file for language detection")

        audio_file = get_audio_file(audio_file_path)
        whisper = WhisperASR(
            model_path="jq/whisper-large-v2-multilingual-prompts-corrected"
        )
        processor, model = whisper.setup_model()
        detected_language = whisper.auto_detect_audio_language(
            audio_file, processor, model
        )

        return {"detected_language": detected_language}

    def tts(self, job_input):
        import base64
        import datetime
        import io
        import os
        import tempfile
        import uuid

        import soundfile as sf
        from pydub import AudioSegment

        from gcp_storage_utils import upload_audio_file_to_gcs
        from spark_tts.tts_utils import SparkTTS

        text = job_input.get("text")
        if not text:
            raise ValueError("Missing text for TTS")

        speaker_id = job_input.get("speaker_id", 248)
        temperature = job_input.get("temperature", 0.8)
        top_k = job_input.get("top_k", 50)
        top_p = job_input.get("top_p", 1.0)
        max_new_audio_tokens = job_input.get("max_new_audio_tokens", 2048)
        normalize = job_input.get("normalize", True)

        tts = SparkTTS()
        wav, sr = tts.text_to_speech(
            text,
            speaker_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_audio_tokens=max_new_audio_tokens,
            normalize=normalize,
        )

        # write WAV to buffer
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.seek(0)

        # convert buffer WAV to MP3 via pydub
        audio = AudioSegment.from_file(buf, format="wav")
        temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio.export(temp_mp3.name, format="mp3")
        temp_mp3.close()

        # upload MP3 to GCS
        bucket = os.getenv("AUDIO_CONTENT_BUCKET_NAME")
        # include UTC timestamp in blob name for uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        blob_name = f"tts/{timestamp}_{uuid.uuid4()}.mp3"
        signed_url = upload_audio_file_to_gcs(
            local_file_path=temp_mp3.name,
            bucket_name=bucket,
            destination_blob_name=blob_name,
            expiration_minutes=30,
        )

        # cleanup temp file
        try:
            os.remove(temp_mp3.name)
        except OSError:
            pass

        return {
            "url": signed_url,
            "blob": blob_name,
            "wav_base64": b64,
            "sample_rate": sr,
        }


def handler(job):
    """Handler function that processes jobs."""
    job_input = job.get("input")
    if not job_input:
        return {"Error": "Job input is missing"}

    task = job_input.get("task")
    if not task:
        return {"Error": "Task is missing from job input"}

    task_handler = TaskHandler()
    task_map = {
        "translate": task_handler.translate,
        "transcribe": task_handler.transcribe,
        "asr_summarise": task_handler.asr_summarise,
        "auto_detect_language": task_handler.auto_detect_language,
        "auto_detect_audio_language": task_handler.auto_detect_audio_language,
        "language_classify": task_handler.language_classify,
        "summarise": task_handler.summarise,
        "tts": task_handler.tts,
    }

    if task not in task_map:
        return {"Error": f"Unknown task: {task}"}

    try:
        return task_map[task](job_input)
    except Exception as e:
        logging.error(f"Error processing task {task}: {e}")
        return {"Error": str(e)}


runpod.serverless.start({"handler": handler})
