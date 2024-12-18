import logging
import math
import os
import warnings
from typing import Tuple

import librosa
import soundfile as sf
import torch
import transformers
from pydub import AudioSegment

warnings.simplefilter(action="ignore", category=FutureWarning)

# Libraries to remove silence and noise from audio
USE_ONNX = False  # change this to True if you want to test onnx model
if USE_ONNX:
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=True,
        onnx=USE_ONNX,
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
else:
    from silero_vad import (
        VADIterator,
        collect_chunks,
        get_speech_timestamps,
        load_silero_vad,
        read_audio,
        save_audio,
    )

    model = load_silero_vad(onnx=USE_ONNX)

SAMPLING_RATE = 16000


def load_audio_resample(audio_file_path, sr=SAMPLING_RATE):
    """Loads an audio file, resamples it, and writes it to a new file.

    Args:
        audio_file_path: Path to the original audio file.
        sr: Target sampling rate (default is SAMPLING_RATE).

    Returns:
        new_file_path: The path to the new audio file with resampled audio.
    """
    # Load the original audio
    audio, orig_sr = librosa.load(audio_file_path)

    # Resample if the original sample rate is different from the target sample rate
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    # Automatically generate a new file path by appending "_resampled" before the extension
    base, ext = os.path.splitext(audio_file_path)
    new_file_path = f"{base}_resampled{ext}"

    # Save the resampled audio to the new file path
    sf.write(new_file_path, audio, sr)

    # Return the path to the new audio file
    return new_file_path


def remove_audio_silence(audio_file_path, output_file_path=None):
    # load_audio_and_resample(audio_file_path, sr=SAMPLING_RATE)
    wav = read_audio(
        load_audio_resample(audio_file_path, sr=SAMPLING_RATE),
        sampling_rate=SAMPLING_RATE,
    )
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
    # pprint(speech_timestamps)
    # merge all speech chunks to one audio
    file_name, ext = os.path.splitext(audio_file_path)
    file_name = os.path.basename(file_name)
    output_file_path = (
        f"{file_name}_only_speech{ext}"
        if output_file_path is None
        else output_file_path
    )
    save_audio(
        output_file_path,
        collect_chunks(speech_timestamps, wav),
        sampling_rate=SAMPLING_RATE,
    )
    return output_file_path


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


class WhisperASR:
    """
    A class for setting up and using a Whisper-based Automatic Speech Recognition (ASR) system.

    This class provides methods for:
    - Setting up the Whisper model and processor.
    - Retrieving language codes based on long or short language names.
    - Creating a pipeline for automatic speech recognition.
    - Loading and resampling audio files.
    - Transcribing audio files using the configured pipeline.

    Example usage:

    # Initialize the class with a model path
    asr_system = WhisperASR('jq/whisper-large-v2-multilingual')

    # Set up the model and processor
    processor, model = asr_system.setup_model()

    # Get the language code for Luganda
    language_code = asr_system.get_language_code('Luganda', processor)

    # Set up the ASR pipeline
    pipeline = asr_system.setup_pipeline(model, processor, language_code=language_code, batch=True)

    # Transcribe an audio file
    transcription = asr_system.transcribe_audio('path_to_audio_file.wav', pipeline, return_timestamps=True)
    """

    def __init__(self, model_path: str = "jq/whisper-large-v2-multilingual"):
        """
        Initializes the WhisperASR class with the specified model path.

        Parameters:
        model_path (str): The path or identifier of the pre-trained model (e.g., a Hugging Face model ID).
        """
        self.model_path = model_path
        try:
            self.device = torch.device("cuda")
        except Exception:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_model(self):
        """
        Sets up and returns a Whisper model and processor for automatic speech recognition.

        Returns:
        tuple: A tuple containing the processor and the model.
            - processor (transformers.WhisperProcessor): The processor used for tokenization and feature extraction.
            - model (transformers.WhisperForConditionalGeneration): The model used for generating text from speech.
        """
        processor = transformers.WhisperProcessor.from_pretrained(
            self.model_path, language=None, task="transcribe"
        )
        model = transformers.WhisperForConditionalGeneration.from_pretrained(
            self.model_path
        ).to(self.device)
        return processor, model

    def get_language_code(self, language: str, processor) -> str:
        """
        Returns the correct language code for a given language using the provided processor.

        Parameters:
        language (str): The name or code of the language (e.g., "English", "eng", "Luganda", "lug", etc.).
        processor: An object that contains a tokenizer used to decode language ID tokens.

        Returns:
        str: The corresponding language code.

        Raises:
        ValueError: If the language is not supported.
        """
        language_codes = {
            "English": "eng",
            "Luganda": "lug",
            "Runyankole": "nyn",
            "Acholi": "ach",
            "Ateso": "teo",
            "Lugbara": "lgg",
            "Swahili": "swa",
            "Kinyarwanda": "kin",
            "Lusoga": "xog",
            "Lumasaba": "myx"
        }

        code_to_language = {v: k for k, v in language_codes.items()}
        standardized_language = (
            language.capitalize() if len(language) > 3 else language.lower()
        )

        if standardized_language in language_codes:
            code = language_codes[standardized_language]
        elif standardized_language in code_to_language:
            code = standardized_language
        else:
            raise ValueError(f"Language '{language}' is not supported.")

        language_id_tokens = {
            "eng": 50259,
            "ach": 50357,
            "lgg": 50356,
            "lug": 50355,
            "nyn": 50354,
            "teo": 50353,
            'xog': 50352,
            'kin': 50350,
            'myx': 50349,
            'swa': 50318,
        }

        token = language_id_tokens[code]
        language_code = processor.tokenizer.decode(token)[2:-2]

        return language_code

    def setup_pipeline(
        self, model, processor, language_code: str, batch: bool
    ) -> transformers.pipeline:
        """
        Creates and returns a Whisper pipeline for automatic speech recognition.

        Parameters:
        model: The pre-trained model to be used in the pipeline.
        processor: A processor object that includes tokenizer and feature extractor.
        language_code (str): The language code for the target language.
        batch (bool): If True, enables batching for the pipeline; otherwise, disables it.

        Returns:
        transformers.pipeline: A configured Whisper pipeline ready for use in automatic speech recognition.
        """
        model_kwargs = (
            {"attn_implementation": "flash_attention_2"}
            if transformers.utils.is_flash_attn_2_available()
            else {"attn_implementation": "sdpa"}
        )

        generate_kwargs = {
            "language": language_code,
            "forced_decoder_ids": None,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 5,
            "num_beams": 3,
        }

        whisper_pipeline = transformers.pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=f"{self.device}:0",
            torch_dtype=torch.float16,
            model_kwargs=model_kwargs,
            generate_kwargs=generate_kwargs,
            **({"batch_size": 2} if batch else {}),
        )

        return whisper_pipeline

    def setup_pipeline_whisper_large_v2_multilingual_prompts_corrected(self):
        """
        Set up the Whisper automatic speech recognition pipeline.

        This method configures and returns a Whisper pipeline using the model
        'jq/whisper-large-v2-multilingual-prompts-corrected"'. The pipeline is set up to transcribe
        audio input using specific configurations like repetition penalty, no repeat n-grams,
        and temperature.

        Returns:
            transformers.Pipeline: Configured Whisper ASR pipeline for transcription.
        """
        whisper_pipeline = transformers.pipeline(
            task="automatic-speech-recognition",
            model=self.model_path,
            device=self.device,
            torch_dtype=torch.float16,
            model_kwargs=({"attn_implementation": "sdpa"}),  # Maybe a speedup?
        )
        return whisper_pipeline

    def auto_detect_audio_language(self, audio_file_path, processor, model):
        audio = self.load_audio_and_resample(audio_file_path=audio_file_path)
        salt_whisper_language_id_tokens = {
            "eng": 50259,
            "ach": 50357,
            "lgg": 50356,
            "lug": 50355,
            "nyn": 50354,
            "teo": 50353,
            'xog': 50352,
            'kin': 50350,
            'myx': 50349,
            'swa': 50318,
        }
        token_to_language = {}
        for lang, token in salt_whisper_language_id_tokens.items():
            token_to_language[token] = lang

        input_features = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            do_normalize=True,
            device=self.device,
        ).input_features
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features.to(self.device), max_new_tokens=10
            )[0]

        language_token = int(predicted_ids[1])

        detected_language = token_to_language.get(language_token, None)

        return detected_language

    def generate_transcribe_kwargs(self, whisper_pipeline, device):
        """
        Generates keyword arguments for the transcription task using a Whisper pipeline.

        This function generates a set of `generate_kwargs` to be used for speech transcription
        by preparing prompt tokens (via the Whisper tokenizer) and setting other task-specific
        parameters for transcription generation. It is tailored for a scenario where the transcription
        task involves a predefined prompt (related to dfcu Bank) and specific settings like beam search.

        Args:
            whisper_pipeline (transformers.Pipeline): The Whisper pipeline object that includes the
                tokenizer and the model for speech recognition.
            device (torch.device or str): The device (e.g., 'cpu' or 'cuda') to which the prompt tokens
                should be moved for model processing.

        Returns:
            dict: A dictionary containing the arguments for speech transcription, including:
                - 'prompt_ids' (torch.Tensor): Tokenized prompt IDs from the provided text.
                - 'prompt_condition_type' (str): Specifies how the prompt should be used (e.g., 'first-segment').
                - 'condition_on_prev_tokens' (bool): Whether to condition the generation on previous tokens.
                - 'task' (str): The task type, set to 'transcribe'.
                - 'language' (NoneType): Language specification for transcription (None implies automatic detection).
                - 'num_beams' (int): Number of beams for beam search during generation (set to 1 for greedy decoding).
        """
        prompt_ids = whisper_pipeline.tokenizer.get_prompt_ids(
            "dfcu, Quick Banking app, QuickApp, Quick Online, Quick Banking platform, "
            "dfcu Personal Banking, mobile app, App store, Google Play Store, "
            "dfcu Quick Online, Quick Connect, internet banking, mobile banking, "
            "smartphone, national ID, passport, trust factor, Pinnacle Current Account,"
            " dfcu SACCO account, savings account, Dembe account, Smart Plan account, "
            "Campus Plus account, Young Savers account, investment club account, "
            "joint account, Secondary Account Ku-Spot, personal loan, mobi loan, save "
            "for loan, home loan, agent banking, banking security, "
            "6th Street, Abayita Ababiri, Bugolobi, Bwaise, Entebbe Road, Impala, "
            "Jinja Road, Kampala Road, Kawempe, Kikuubo, Kireka, Kyadondo, Kyambogo, "
            "Lugogo, Makerere, Market Street, Naalya, Nabugabo, Sun City, Acacia, "
            "Entebbe Town, Kyengera, Luwum Street, Nateete, Ndeeba, Nsambya, Ntinda "
            "Shopping Centre (Capital Shoppers), Ntinda Trading Centre, Owino, "
            "William Street, Abim, Arua, Dokolo, Gulu, Hoima, Ibanda, Iganga, Ishaka, "
            "Isingiro, Jinja, Kabale, Kisoro, Kitgum, Lira, Luweero, Lyantonde, "
            "Masaka, Mbale, Mbarara, Mukono, Ntungamo, Pader, Pallisa, Rushere, "
            "Soroti, Tororo. "
            "Thank you for calling dfcu bank. How can I help you? ",
            return_tensors="pt",
        ).to(device)

        generate_kwargs = {
            "prompt_ids": prompt_ids,
            "prompt_condition_type": "first-segment",
            "condition_on_prev_tokens": True,
            "task": "transcribe",
            "language": None,
            "no_repeat_ngram_size": 5,
            "num_beams": 1,
        }

        return generate_kwargs

    def load_audio_and_resample(self, audio_file_path, sr=16000):
        """
        Loads an audio file and resamples it to the desired sample rate.

        Parameters:
        audio_file_path (str): Path to the audio file.
        sr (int): Target sample rate. Default is 16000.

        Returns:
        numpy.ndarray: The resampled audio data.
        """
        audio, orig_sr = librosa.load(audio_file_path)
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        return audio

    def transcribe_audio(
        self, audio_file_path: str, pipeline, return_timestamps: bool = False
    ):
        """
        Transcribes an audio file using the Whisper pipeline.

        Parameters:
        audio_file_path (str): Path to the audio file.
        pipeline: The configured Whisper pipeline for automatic speech recognition.
        return_timestamps (bool): Whether to return timestamps in the transcription.

        Returns:
        dict: The transcription result, including the text and optionally timestamps.
        """
        filename = os.path.basename(audio_file_path)
        logging.info(f"File name: {filename}")
        file_size_mb, duration_minutes = get_audio_file_info(audio_file_path)
        logging.info(
            f"File size: {file_size_mb}mb Duration in minutes: {duration_minutes}"
        )

        if math.floor(duration_minutes) > 1:
            # Get the transcription using the pipeline
            transcription = pipeline(
                audio_file_path, return_timestamps=return_timestamps
            )
        else:
            # Load and resample the audio
            audio = self.load_audio_and_resample(audio_file_path)
            # Get the transcription using the pipeline
            transcription = pipeline(audio, return_timestamps=return_timestamps)

        return transcription


if __name__ == "__main__":
    model_id = "jq/whisper-large-v2-salt-plus-xog-myx-kin-swa-sample-packing"
    language = "ach"
    asr_system = WhisperASR(model_id)
    # Set up the model and processor by calling setup_model()
    processor, model = asr_system.setup_model()
    # Retrieve the language code by calling get_language_code() with the language name or code.
    language_code = asr_system.get_language_code(language, processor)
    # Set up the ASR pipeline by calling setup_pipeline()
    pipeline = asr_system.setup_pipeline(
        model, processor, language_code=language_code, batch=True
    )

    audio_files = [
        "./content/MEGA 12.2.mp3",
        # "./content/Lutino weng pwonye - Dul 1 - Introduction - Including Radio Maria.mp3",
    ]

    for audio_file in audio_files:
        if os.path.exists(audio_file):
            try:
                transcription = asr_system.transcribe_audio(
                    audio_file, pipeline, return_timestamps=True
                )
                print(
                    f"Transcription for {os.path.basename(audio_file)}: {transcription.get('text')}"
                )
            except Exception as e:
                print(str(e))
                pipeline = asr_system.setup_pipeline(
                    model, processor, language_code=language_code, batch=False
                )
                transcription = asr_system.transcribe_audio(
                    audio_file, pipeline, return_timestamps=True
                )
                print(
                    f"Transcription for {os.path.basename(audio_file)}: {transcription.get('text')}"
                )
        else:
            print(f"File {audio_file} does not exist.")
