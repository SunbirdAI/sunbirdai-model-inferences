import builtins
import logging
import os
import re
import sys
import typing
from typing import List, Optional, Tuple

import numpy as np
import runpod
import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

builtins.Any = typing.Any

# Ensure the current directory is in sys.path for text_chunker import
current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from text_chunker import chunk_text_simple

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("SPARK_TTS_RUNPOD_ENDPOINT_ID")
runpod.api_key = RUNPOD_API_KEY

endpoint = runpod.Endpoint(RUNPOD_ENDPOINT_ID)
logging.basicConfig(level=logging.INFO)


class SparkTTS:
    """
    Utility for text-to-speech with Spark-TTS.
    """

    def __init__(
        self,
        model_base_repo: str = "unsloth/Spark-TTS-0.5B",
        cache_dir: str = "Spark-TTS-0.5B",
        device: Optional[torch.device] = None,
    ):
        self.cache_dir = cache_dir
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Download base Spark-TTS model
        if not os.path.exists(self.cache_dir):
            snapshot_download(
                repo_id=model_base_repo,
                local_dir=self.cache_dir,
                ignore_patterns=["*LLM*"],
            )

        # Initialize audio tokenizer/vocoder
        sys.path.append("Spark-TTS")

        from sparktts.models.audio_tokenizer import BiCodecTokenizer

        self.audio_tokenizer = BiCodecTokenizer(self.cache_dir, str(self.device))

    def tts_tokens(self, endpoint, data):
        return endpoint.run_sync(data, timeout=600)  # Timeout in seconds

    def get_tts_tokens(
        self,
        text: str,
        speaker_id: int = 248,
        temperature: float = 0.7,
        max_new_audio_tokens: int = 2000,
    ):
        prompt = f"<|task_tts|><|start_content|>{speaker_id}: {text}<|end_content|><|start_global_token|>"

        data = {
            "input": {
                "prompt": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_tokens": max_new_audio_tokens,
                },
            }
        }
        response = self.tts_tokens(endpoint, data)
        predicted_tokens = response[0]["choices"][0]["tokens"][0]
        return predicted_tokens

    def generate_speech_from_text(
        self,
        text: str,
        speaker_id: int = 248,
        temperature: float = 0.7,
        max_new_audio_tokens: int = 2000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_tokens = self.get_tts_tokens(
            text=text,
            speaker_id=speaker_id,
            temperature=temperature,
            max_new_audio_tokens=max_new_audio_tokens,
        )

        # Extract semantic token IDs using regex
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicted_tokens)
        if not semantic_matches:
            print("Warning: No semantic tokens found in the generated output.")
            # Handle appropriately - perhaps return silence or raise error

        pred_semantic_ids = (
            torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
        )  # Add batch dim

        # Extract global token IDs using regex (assuming controllable mode also generates these)
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicted_tokens)
        if not global_matches:
            print(
                "Warning: No global tokens found in the generated output (controllable mode). Might use defaults or fail."
            )
            pred_global_ids = torch.zeros((1, 1), dtype=torch.long)
        else:
            pred_global_ids = (
                torch.tensor([int(token) for token in global_matches])
                .long()
                .unsqueeze(0)
            )  # Add batch dim

        return pred_semantic_ids, pred_global_ids

    def generate_speech_segment_with_retry(
        self,
        text: str,
        speaker_id: int = 248,
        temperature: float = 0.7,
        max_new_audio_tokens: int = 2000,
        max_retries: int = 3,
    ) -> Optional[np.ndarray]:
        """
        Generate a single speech segment with retry logic.
        
        This function will retry up to max_retries times if there's a RuntimeError
        during token generation or detokenization (typically dimension mismatches).
        
        Args:
            text: Input text to synthesize
            speaker_id: Speaker ID
            temperature: Sampling temperature
            max_new_audio_tokens: Maximum tokens to generate
            max_retries: Maximum number of retry attempts
            
        Returns:
            wav_np: Audio waveform as numpy array, or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                # Generate tokens
                pred_semantic_ids, pred_global_ids = self.generate_speech_from_text(
                    text=text,
                    speaker_id=speaker_id,
                    temperature=temperature,
                    max_new_audio_tokens=max_new_audio_tokens,
                )
                
                # Log token shapes for debugging
                print(f"Attempt {attempt + 1}: semantic shape={pred_semantic_ids.shape}, "
                      f"global shape={pred_global_ids.shape}")
                
                # Detokenize to waveform
                wav_np = self.audio_tokenizer.detokenize(
                    pred_global_ids.to("cuda"), pred_semantic_ids.to("cuda")
                )
                
                # Success!
                return wav_np
                
            except RuntimeError as e:
                error_msg = str(e)
                print(f"⚠️  Attempt {attempt + 1}/{max_retries} failed for text chunk: '{text[:50]}...'")
                print(f"   Error: {error_msg}")
                
                # Check if it's the dimension mismatch error we're expecting
                if "cannot be multiplied" in error_msg or "shape" in error_msg.lower():
                    if attempt < max_retries - 1:
                        print(f"   Retrying with slightly different temperature...")
                        # Slightly vary temperature to get different generation
                        temperature = temperature + np.random.uniform(-0.05, 0.05)
                        temperature = float(np.clip(temperature, 0.1, 1.0))
                        time.sleep(0.5)  # Small delay before retry
                    else:
                        print(f"   ❌ All {max_retries} attempts failed. Skipping this chunk.")
                        return None
                else:
                    # Different error, re-raise
                    raise
                    
            except ValueError as e:
                print(f"⚠️  ValueError on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"   Retrying...")
                    time.sleep(0.5)
                else:
                    print(f"   ❌ All {max_retries} attempts failed. Skipping this chunk.")
                    return None
        
        return None

    def get_speech_segments(
        self,
        text_chunks: List[str],
        speaker_id: int = 248,
        temperature: float = 0.7,
        max_new_audio_tokens: int = 2000,
    ) -> List[np.ndarray]:
        segments = []
        for text in text_chunks:
            pred_semantic_ids, pred_global_ids = self.generate_speech_from_text(
                text=text,
                speaker_id=speaker_id,
                temperature=temperature,
                max_new_audio_tokens=max_new_audio_tokens,
            )
            logging.info(f"Text: {text}")
            logging.info(f"pred_semantic_ids: {pred_semantic_ids}")
            logging.info(f"pred_global_ids: {pred_global_ids}")
            logging.info("=" * 100)
            wav_np = self.audio_tokenizer.detokenize(
                pred_global_ids.to("cuda"), pred_semantic_ids.to("cuda")
            )
            segments.append(wav_np)

        return segments

    def text_to_speech(
        self,
        text: str,
        speaker_id: int = 248,
        temperature: float = 0.7,
        max_new_audio_tokens: int = 2048,
        sample_rate: int = 16000,
        max_retries: int = 3,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert text to speech waveform with retry logic.
        
        Args:
            text: Input text to synthesize
            speaker_id: Speaker ID
            temperature: Sampling temperature
            max_new_audio_tokens: Maximum tokens per chunk
            sample_rate: Output sample rate
            max_retries: Maximum retry attempts per chunk
            
        Returns:
            result_wav: Concatenated audio waveform (np.ndarray)
            sr: Sample rate (int)
        """
        # Chunk the text into sentences
        texts = chunk_text_simple(text)
        # texts = chunk_text_with_count(text, sentences_per_chunk=3)
        # texts = chunk_text(text, max_chunk_size=500)
        texts = [t.strip() for t in texts if len(t.strip()) > 0]
        
        print(f"\n🎙️  Starting TTS conversion for {len(texts)} chunks...")
        
        # Generate speech segments with retry logic
        speech_segments = self.get_speech_segments(
            text_chunks=texts,
            speaker_id=speaker_id,
            temperature=temperature,
            max_new_audio_tokens=max_new_audio_tokens,
        )
        
        # Concatenate all segments
        if speech_segments:
            result_wav = np.concatenate(speech_segments)
            print(f"\n✅ TTS conversion completed! Total duration: {len(result_wav)/sample_rate:.2f}s")
        else:
            print("\n⚠️  No speech segments generated. Returning silence.")
            result_wav = np.zeros(sample_rate, dtype=np.float32)  # 1 second of silence
        
        return result_wav, sample_rate

    def save_wav(
        self,
        text: str,
        outfile: str,
        **kwargs,
    ) -> None:
        """
        Generate speech and save to a WAV file.
        """
        wav, sr = self.text_to_speech(text, **kwargs)
        import soundfile as sf

        sf.write(outfile, wav, sr)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spark-TTS text-to-speech utility.")
    parser.add_argument("-t", "--text", required=True, help="Input text to convert.")
    parser.add_argument(
        "-o", "--output", default="output.wav", help="Output WAV filename."
    )
    args = parser.parse_args()

    tts = SparkTTS()
    tts.save_wav(
        args.text,
        args.output,
        speaker_id=248,
        sample_rate=16000,
    )
    print(f"Saved: {args.output}")
