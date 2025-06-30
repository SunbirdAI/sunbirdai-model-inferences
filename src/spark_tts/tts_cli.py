from typing import Optional

import typer
from tts_utils import SparkTTS

cli = typer.Typer()


@cli.command()
def tts(
    text: str = typer.Option(..., help="Text to convert to speech."),
    speaker_id: str = typer.Option(248, help="Speaker Language Id."),
    sample_rate: int = typer.Option(16000, help="Audio sample rate."),
    output: str = typer.Option("output.wav", help="Output WAV file path."),
    no_normalize: bool = typer.Option(True, help="Disable volume normalization."),
    adapter_repo: Optional[str] = typer.Option(
        None, help="HF adapter repo (optional)."
    ),
    adapter_filename: Optional[str] = typer.Option(
        None, help="Adapter filename (optional)."
    ),
):
    """Generate speech from text and save to WAV."""
    tts_engine = SparkTTS(adapter_repo=adapter_repo, adapter_filename=adapter_filename)
    tts_engine.save_wav(text, output, normalize=not no_normalize, speaker_id=speaker_id, sample_rate=sample_rate)
    typer.echo(f"Saved WAV to {output}")


if __name__ == "__main__":
    cli()
