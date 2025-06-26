def audio_file_to_bytes(file_path: str) -> bytes:
    """
    Read the entire contents of an audio file and return it as raw bytes.

    Args:
        file_path: Path to the audio file to read (e.g., WAV, MP3).

    Returns:
        A bytes object containing the raw file data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If there is an error reading the file.
    """
    with open(file_path, "rb") as f:
        return f.read()


def bytes_to_audio_file(data: bytes, output_path: str) -> None:
    """
    Write raw audio bytes to a file on disk.

    Args:
        data: Raw audio file data as bytes (e.g., WAV or MP3 bytes).
        output_path: Destination path where the audio file will be written.

    Returns:
        None. On success, the file at output_path will contain the audio data.

    Raises:
        IOError: If there is an error writing to the output file.
    """
    with open(output_path, "wb") as f:
        f.write(data)


if __name__ == "__main__":
    # Example usage
    input_file = "/Users/patrickwalukagga/Projects/Sunbirdai/sunbirdai-model-inferences/content/Acholi 0.mp3"  # Replace with your audio file path
    output_file = "/Users/patrickwalukagga/Projects/Sunbirdai/sunbirdai-model-inferences/content/output.wav"  # Replace with desired output file path

    # Read audio file to bytes
    audio_bytes = audio_file_to_bytes(input_file)
    print(f"Read {len(audio_bytes)} bytes from {input_file}")
    # Write bytes to a text file (optional, for debugging)
    with open("audio_bytes.txt", "wb") as byte_file:
        byte_file.write(audio_bytes)
    print("Wrote audio bytes to audio_bytes.txt for debugging.")

    # Write bytes back to a new audio file
    bytes_to_audio_file(audio_bytes, output_file)
    print(f"Wrote audio data to {output_file}")
