import os
from typing import Optional, Union

import librosa
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()

SAMPLE_RATE = 16000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiarizationPipeline:
    """
    A pipeline for performing speaker diarization on audio data.

    This class initializes with a pretrained diarization model and can be called
    with an audio file or waveform to perform diarization, returning a DataFrame
    with the start and end times for each speaker segment.

    Attributes:
        model (Pipeline): The loaded diarization model ready for inference.
    """

    def __init__(
        self,
        model_name="pyannote/speaker-diarization-3.0",
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        """
        Initializes the DiarizationPipeline with a pretrained model.

        Args:
            model_name (str): The name of the pretrained diarization model to load.
            use_auth_token (str, optional): Token to use for authentication if the model
                                            is from a private repository. Defaults to None.
            device (str or torch.device, optional): The device on which to run the model,
                                                    either "cpu" or "cuda". Defaults to "cpu".
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.model = Pipeline.from_pretrained(
            model_name, use_auth_token=use_auth_token
        ).to(device)

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Perform diarization on the provided audio.

        Args:
            audio (str or np.ndarray): The path to the audio file or a numpy array of the waveform.
            min_speakers (int, optional): The minimum number of speakers to assume in the diarization
                                          process. Defaults to None.
            max_speakers (int, optional): The maximum number of speakers to assume in the diarization
                                          process. Defaults to None.

        Returns:
            DataFrame: A pandas DataFrame with columns for the segment, label, speaker,
                       start time, and end time of each speaker segment.
        """
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": SAMPLE_RATE,
        }
        segments = self.model(
            audio_data, min_speakers=min_speakers, max_speakers=max_speakers
        )
        diarize_df = pd.DataFrame(
            segments.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
        return diarize_df


class Segment:
    """
    A class to represent a single segment of audio with a start time, end time, and speaker label.

    This class is typically used to encapsulate the information about a segment of audio that
    has been identified during a speaker diarization process, including the time the segment
    starts, when it ends, and which speaker is speaking.

    Attributes:
        start (float): The start time of the audio segment in seconds.
        end (float): The end time of the audio segment in seconds.
        speaker (str, optional): The label of the speaker for this audio segment. Defaults to None.
    """

    def __init__(self, start, end, speaker=None):
        """
        Initializes a new instance of the Segment class.

        Args:
            start (float): The start time of the audio segment in seconds.
            end (float): The end time of the audio segment in seconds.
            speaker (str, optional): The label of the speaker for this segment. If not specified,
                                     the speaker attribute is set to None.
        """
        self.start = start
        self.end = end
        self.speaker = speaker


def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Open an audio file and read as mono waveform, resampling as necessary.

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # librosa automatically resamples to the given sample rate (if necessary)
        # and converts the signal to mono (by averaging channels)
        audio, _ = librosa.load(file, sr=sr, mono=True, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio with librosa: {e}") from e

    return audio


def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    """
    Assign speakers to segments of a transcript based on the results of a diarization data frame.

    This function iterates through segments of a transcript and assigns the speaker labels
    based on the overlap between the speech segments and the diarization data. Optionally,
    if `fill_nearest` is True, the function will assign speakers even to segments that
    do not have a direct overlap with the diarization data by finding the closest speaker
    in time.

    Parameters:
    - diarize_df (DataFrame): A pandas DataFrame containing the diarization information
                              with columns 'start', 'end', and 'speaker'.
    - transcript_result (dict): A dictionary with a key 'chunks' that contains a list of
                                transcript segments, where each segment is a dictionary
                                with keys 'text' and 'timestamp' (a tuple with start and end times).
    - fill_nearest (bool, optional): A flag to determine whether to assign speakers to all segments
                                     based on the nearest speaker data if no direct overlap is found.
                                     Defaults to False.

    Returns:
    - dict: The updated transcript_result with speakers assigned to each segment.

    Examples of diarize_df and transcript_result structures:

    diarize_df example:
        speaker  start   end
        0        0.0     1.5
        1        1.5     3.0

    transcript_result example:
        {'chunks': [{'text': 'Hello', 'timestamp': (0.5, 1.0)},
                    {'text': 'world', 'timestamp': (1.5, 2.0)}]}

    Example usage:
    >>> diarize_df = pd.DataFrame({'speaker': [0, 1], 'start': [0.0, 1.5], 'end': [1.5, 3.0]})
    >>> transcript_result = {'chunks': [{'text': 'Hello', 'timestamp': (0.5, 1.0)},
                                         {'text': 'world', 'timestamp': (1.5, 2.0)}]}
    >>> assign_word_speakers(diarize_df, transcript_result)
    {'chunks': [{'text': 'Hello', 'timestamp': (0.5, 1.0), 'speaker': 0},
                {'text': 'world', 'timestamp': (1.5, 2.0), 'speaker': 1}]}
    """
    transcript_segments = transcript_result["chunks"]

    for seg in transcript_segments:
        # Calculate intersection and union between diarization segments and transcript segment
        diarize_df["intersection"] = np.minimum(
            diarize_df["end"], seg["timestamp"][1]
        ) - np.maximum(diarize_df["start"], seg["timestamp"][0])
        diarize_df["union"] = np.maximum(
            diarize_df["end"], seg["timestamp"][1]
        ) - np.minimum(diarize_df["start"], seg["timestamp"][0])

        # Filter out diarization segments with no overlap if fill_nearest is False
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df["intersection"] > 0]
        else:
            dia_tmp = diarize_df

        # If there are overlapping segments, assign the speaker with the greatest overlap
        if len(dia_tmp) > 0:
            speaker = (
                dia_tmp.groupby("speaker")["intersection"]
                .sum()
                .sort_values(ascending=False)
                .index[0]
            )
            seg["speaker"] = speaker

    return transcript_result


def process_audio_diarization(audio_file, hf_token, transcription, device):
    """
    Process an audio file for speaker diarization and assign speakers to the transcription.

    Args:
        audio_file (str): Path to the audio file to be processed.
        hf_token (str): Hugging Face authentication token for accessing the diarization model.
        transcription (str): Transcription of the audio file.
        device (str): Device to run the model on, either 'cuda' for GPU or 'cpu' for CPU.

    Returns:
        dict: A dictionary mapping words in the transcription to the identified speakers.

    Example:
        audio_file = "./content/kitakas_eng.mp3"
        hf_token = "your_hf_token"
        transcription = "transcribed text from audio"
        device = "cuda"
        output = process_audio_diarization(audio_file, hf_token, transcription, device)
        print(output)

    Raises:
        ValueError: If any of the arguments are invalid or if the diarization process fails.
    """
    # Initialize the diarization model
    diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)

    # Perform diarization on the audio file
    diarize_segments = diarize_model(audio_file, min_speakers=1, max_speakers=2)

    # Assign speakers to the transcription
    output = assign_word_speakers(diarize_segments, transcription)

    return output


def format_diarization_output(diarization_output):
    """
    Format the diarization output into sentences with timestamps and assigned speakers in their sequential order of speech.

    Args:
        diarization_output (dict): The output dictionary containing 'text' and 'chunks'.
            The 'chunks' are expected to have 'text', 'timestamp', and 'speaker' keys.

    Returns:
        str: Formatted string representing the diarization with speakers, timestamps, and sentences.
    """
    chunks = diarization_output["chunks"]
    formatted_output = []
    current_speaker = None
    current_sentence = []
    current_start_time = None

    for chunk in chunks:
        word = chunk["text"]
        start_time, end_time = chunk["timestamp"]
        speaker = chunk["speaker"]

        # Handle change of speaker
        if current_speaker is not None and speaker != current_speaker:
            sentence_text = " ".join(current_sentence)
            formatted_output.append(f"**{current_speaker}**")
            formatted_output.append(
                f"({current_start_time:.2f}) {sentence_text} ({end_time:.2f})"
            )
            current_sentence = []

        # Add word to the current sentence
        current_sentence.append(word)

        # Update the current speaker and start time
        if current_speaker != speaker:
            current_speaker = speaker
            current_start_time = start_time

    # Add the last sentence
    if current_sentence:
        sentence_text = " ".join(current_sentence)
        formatted_output.append(f"**{current_speaker}**")
        formatted_output.append(
            f"({current_start_time:.2f}) {sentence_text} ({end_time:.2f})"
        )

    return "\n".join(formatted_output)


if __name__ == "__main__":
    from constants import BACKUP_SLICED_TRANSCRIPTION, KITAKAS_TRANSCRIPTION

    results = []

    transcriptions = {
        "backup": {
            "transcription": BACKUP_SLICED_TRANSCRIPTION,
            "audio_file": "./content/backup_sliced.mp3",
        },
        "kitaakas": {
            "transcription": KITAKAS_TRANSCRIPTION,
            "audio_file": "./content/kitakas_eng.mp3",
        },
    }

    for key in transcriptions:
        transcription = transcriptions[key].get("transcription")
        tmp_results = transcription["text"]
        audio_file = transcriptions[key].get("audio_file")
        hf_token = os.getenv("HF_TOKEN")
        diarization_output = process_audio_diarization(
            audio_file, hf_token, transcription, device
        )
        print(diarization_output)
        print("=" * 100)
        formatted_diarization_output = format_diarization_output(diarization_output)
        print(formatted_diarization_output)
        print("=" * 100)

    """Example Formatted output
    **SPEAKER_01**
    (2.42) this is the chataka's broadcast my husband and i will be letting in honor life as a couple (8.34)
    **SPEAKER_00**
    (8.06) husband and helper (9.68)
    **SPEAKER_01**
    (9.36) husband and wife as a (12.80)
    **SPEAKER_00**
    (12.48) marriage is not a new wild you enter into you don't become a new person you come with what you been working on (19.54)
    **SPEAKER_01**
    (19.42) it's easy to go through the first year of your marriage trying to knit pick the shortcomings of your partner now this 
    is our first episode and it's a series of random reflections from our one year in marriage now we hope that as we share experiences 
    and insights on our journey the you will be inspired to pursue the portion and purpose to your marriage so this is the chitaka's 
    podcast and these are random reflections when you are married (45.92)
    """
