import os
import re
import shutil
import sys
from typing import Optional, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from unsloth import FastModel

# Ensure the current directory is in sys.path for text_chunker import
current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from text_chunker import chunk_text


class SparkTTS:
    """
    Utility for text-to-speech with Spark-TTS.
    """

    def __init__(
        self,
        model_base_repo: str = "unsloth/Spark-TTS-0.5B",
        adapter_repo: Optional[str] = None,
        adapter_filename: Optional[str] = None,
        cache_dir: str = "Spark-TTS-0.5B",
        device: Optional[torch.device] = None,
    ):
        self.cache_dir = cache_dir
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Download base Spark-TTS model
        if not os.path.exists(self.cache_dir):
            snapshot_download(repo_id=model_base_repo, local_dir=self.cache_dir)

        # Optionally download and overwrite with adapter
        if adapter_repo and adapter_filename:
            ckpt = hf_hub_download(repo_id=adapter_repo, filename=adapter_filename)
            target = os.path.join(self.cache_dir, "LLM", adapter_filename)
            shutil.copy(ckpt, target)

        # Load the FastModel
        model_path = os.path.join(self.cache_dir, "LLM")
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=torch.float32,
            full_finetuning=False,
            load_in_4bit=False,
        )
        FastModel.for_inference(self.model)
        self.model.to(self.device)

        # Initialize audio tokenizer/vocoder
        sys.path.append("Spark-TTS")

        from sparktts.models.audio_tokenizer import BiCodecTokenizer

        self.audio_tokenizer = BiCodecTokenizer(self.cache_dir, str(self.device))

    @torch.inference_mode()
    def generate_tokens(
        self,
        text: str,
        speaker_id: int = 242,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_audio_tokens: int = 2048,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates bicodec global and semantic tokens from input text.
        Returns:
            pred_global_ids: torch.Tensor
            pred_semantic_ids: torch.Tensor
        """
        # Control prompt prefix for voice
        text = f"{speaker_id}: {text}"
        prompt = "".join(
            [
                "<|task_tts|>",
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
            ]
        )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_audio_tokens,  # Limit generation length
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,  # Stop token
            pad_token_id=self.tokenizer.pad_token_id,  # Use models pad token id
        )
        generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1] :]
        predicts_text = self.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False
        )[0]

        # Extract semantic token IDs using regex
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
        if not semantic_matches:
            print("Warning: No semantic tokens found in the generated output.")
            # Handle appropriately - perhaps return silence or raise error
            return np.array([], dtype=np.float32)

        pred_semantic_ids = (
            torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
        )  # Add batch dim

        # Extract global token IDs using regex (assuming controllable mode also generates these)
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
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

        pred_global_ids = pred_global_ids.unsqueeze(0)  # Shape becomes (1, 1, N_global)

        return pred_global_ids, pred_semantic_ids

    def text_to_speech(
        self,
        text: str,
        speaker_id: int = 248,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_audio_tokens: int = 2048,
        sample_rate: int = 16000,
        normalize: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert text to speech waveform.
        Returns:
            waveform: np.ndarray (float32)
            sample_rate: int
        """
        texts = chunk_text(text, chunk_size=10)
        texts = [t.strip() for t in texts if len(t.strip()) > 0]
        segments = []
        for text in texts:
            pred_global_ids, pred_semantic_ids = self.generate_tokens(
                text=text,
                speaker_id=speaker_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_audio_tokens=max_new_audio_tokens,
            )
            wav = self.audio_tokenizer.detokenize(
                pred_global_ids.to(self.device).squeeze(0),
                pred_semantic_ids.to(self.device),
            )
            segments.append(wav)
        result_wav = np.concatenate(segments)
        if normalize:
            sys.path.append("Spark-TTS")
            from sparktts.utils.audio import audio_volume_normalize

            result_wav = audio_volume_normalize(result_wav)
        # Default Spark-TTS sample rate
        sr = sample_rate
        return result_wav, sr

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
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable volume normalization.",
    )
    args = parser.parse_args()

    tts = SparkTTS(
        adapter_repo="jq/spark-tts-salt", adapter_filename="model.safetensors"
    )
    tts.save_wav(
        args.text,
        args.output,
        normalize=not args.no_normalize,
        speaker_id=248,
        sample_rate=16000,
    )
    print(f"Saved: {args.output}")
