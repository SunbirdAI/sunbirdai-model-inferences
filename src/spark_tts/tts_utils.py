import os
import sys
import re
import shutil
from typing import Optional, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download, hf_hub_download
from unsloth import FastModel


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
        self.model, _ = FastModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=torch.float32,
            full_finetuning=False,
            load_in_4bit=False,
        )
        FastModel.for_inference(self.model)
        self.model.to(self.device)

        # Initialize audio tokenizer/vocoder
        sys.path.append('Spark-TTS')

        from sparktts.models.audio_tokenizer import BiCodecTokenizer
        self.audio_tokenizer = BiCodecTokenizer(self.cache_dir, str(self.device))

    @torch.inference_mode()
    def generate_tokens(
        self,
        text: str,
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
        input_text = f"242: {text}"
        out = self.model.generate(
            input_text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_audio_tokens,
        )
        output_str = out[0]
        # Extract semantic and global tokens
        semantic_ids = re.findall(r"<\|bicodec_semantic_(\d+?)\|>", output_str)
        global_ids = re.findall(r"<\|bicodec_global_(\d+?)\|>", output_str)
        if not semantic_ids:
            raise RuntimeError("No semantic tokens generated.")

        pred_semantic = torch.tensor([int(i) for i in semantic_ids], dtype=torch.long)
        pred_global = (
            torch.tensor([int(i) for i in global_ids], dtype=torch.long)
            if global_ids
            else torch.zeros(1, dtype=torch.long)
        )
        return pred_global.unsqueeze(0), pred_semantic.unsqueeze(0)

    def text_to_speech(
        self,
        text: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_audio_tokens: int = 2048,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert text to speech waveform.
        Returns:
            waveform: np.ndarray (float32)
            sample_rate: int
        """
        pred_global, pred_semantic = self.generate_tokens(
            text, temperature, top_k, top_p, max_new_audio_tokens
        )
        wav = self.audio_tokenizer.detokenize(
            pred_global.to(self.device).squeeze(0),
            pred_semantic.to(self.device),
        )
        if normalize:
            sys.path.append('Spark-TTS')
            from sparktts.utils.audio_utils import audio_volume_normalize
            wav = audio_volume_normalize(wav)
        # Default Spark-TTS sample rate
        sr = 16000
        return wav, sr

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
        "-o", "--output", default="output.wav", help="Output WAV filename.")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable volume normalization.",
    )
    args = parser.parse_args()

    tts = SparkTTS(adapter_repo="jq/spark-tts-salt", adapter_filename="model.safetensors")
    tts.save_wav(
        args.text, args.output, normalize=not args.no_normalize
    )
    print(f"Saved: {args.output}")
