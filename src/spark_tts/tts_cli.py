import builtins
import typing

import typer
from tts_utils import SparkTTS

builtins.Any = typing.Any

cli = typer.Typer()


@cli.command()
def tts(
    text: str = typer.Option(..., help="Text to convert to speech."),
    speaker_id: str = typer.Option(248, help="Speaker Language Id."),
    sample_rate: int = typer.Option(16000, help="Audio sample rate."),
    output: str = typer.Option("output.wav", help="Output WAV file path."),
):
    """Generate speech from text and save to WAV."""
    tts_engine = SparkTTS()
    tts_engine.save_wav(
        text,
        output,
        speaker_id=speaker_id,
        sample_rate=sample_rate,
    )
    typer.echo(f"Saved WAV to {output}")


if __name__ == "__main__":
    cli()
