import typer
from typing import Optional, Tuple

from tts_utils import SparkTTS

cli = typer.Typer()

@cli.command()
def tts(
    text: str = typer.Option(..., help="Text to convert to speech."),
    output: str = typer.Option("output.wav", help="Output WAV file path."),
    no_normalize: bool = typer.Option(False, help="Disable volume normalization."),
    adapter_repo: Optional[str] = typer.Option(None, help="HF adapter repo (optional)."),
    adapter_filename: Optional[str] = typer.Option(None, help="Adapter filename (optional)."),
):
    """Generate speech from text and save to WAV."""
    tts_engine = SparkTTS(adapter_repo=adapter_repo, adapter_filename=adapter_filename)
    tts_engine.save_wav(text, output, normalize=not no_normalize)
    typer.echo(f"Saved WAV to {output}")

if __name__ == "__main__":
    cli()
