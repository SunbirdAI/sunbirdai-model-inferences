import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import transformers
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())  # Load environment variables from .env file


def create_custom_logger(
    name: str = __name__,
    log_file: str = "app.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Create and configure a logger with both console and file handlers.

    This helper centralizes logger creation so callers can get a consistently
    configured logger across modules. It configures a timestamped formatter,
    writes INFO (and above) to stdout by default and DEBUG (and above) to a
    file. If the named logger already has handlers, new handlers will not be
    added to avoid duplicate logging.

    Parameters
    - name: The logger name (typically __name__).
    - log_file: Path to the logfile where DEBUG+ messages are written.
    - console_level: Logging level for the console handler (default: INFO).
    - file_level: Logging level for the file handler (default: DEBUG).

    Returns
    - logging.Logger: Configured logger instance ready for use.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # capture all levels; handlers filter further

    # Formatter: standard structure
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # StreamHandler (console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # FileHandler (log everything including DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    # Avoid duplicate logs
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# Create module-level logger using the helper
logger = create_custom_logger(__name__)

# Initialize model and tokenizer (move to global scope for reuse)
tokenizer = transformers.NllbTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-1.3B"
)
model = transformers.M2M100ForConditionalGeneration.from_pretrained(
    "jq/nllb-1.3B-many-to-many-pronouncorrection-charaug"
)

try:
    device = torch.device("cuda")
except Exception:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device once
model = model.to(device)
model.eval()  # Set to evaluation mode

# Cache for tokenized inputs to avoid re-tokenization
_tokenization_cache = {}


def make_response(response):
    return {"data": response}


def read_text_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read and return the text contents of a file.

    This helper opens the file at `file_path`, decodes it using `encoding`, and
    returns the resulting string. It logs a helpful error message if the file
    does not exist or another I/O error occurs, then re-raises the exception so
    callers can handle it appropriately.

    Parameters
    - file_path: Path to the file to read. Can be a string or Path-like.
    - encoding: Text encoding to use when reading the file (default: 'utf-8').

    Returns
    - str: The file contents as a string.

    Raises
    - FileNotFoundError: If the file does not exist.
    - UnicodeDecodeError: If the file cannot be decoded using the provided encoding.
    - OSError: For other I/O related errors.
    """

    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        text = path.read_text(encoding=encoding)
        return text
    except Exception as exc:
        logger.exception(f"Failed to read file '{file_path}': {exc}")
        raise


def write_text_file(
    file_path: str, text: str, encoding: str = "utf-8", overwrite: bool = True
) -> None:
    """
    Write text to a file at the specified path.

    This helper writes `text` to `file_path` using the given `encoding`.
    By default it will overwrite existing files. If `overwrite` is set to
    False and the target file already exists, a FileExistsError is raised.

    Parameters
    - file_path: Destination path for the text. Can be a string or Path-like.
    - text: The string content to write to the file.
    - encoding: Text encoding to use when writing (default: 'utf-8').
    - overwrite: If False, do not overwrite an existing file (default: True).

    Returns
    - None

    Raises
    - FileExistsError: If the file exists and overwrite is False.
    - OSError: For other I/O related errors.
    """

    path = Path(file_path)
    if path.exists() and not overwrite:
        logger.error(f"Refusing to overwrite existing file: {file_path}")
        raise FileExistsError(f"File already exists: {file_path}")

    try:
        # Ensure parent dir exists
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(text, encoding=encoding)
        logger.info(f"Wrote text to file: {file_path}")
    except Exception as exc:
        logger.exception(f"Failed to write file '{file_path}': {exc}")
        raise


def get_optimal_max_length(text: str, target_language: str) -> int:
    """
    Calculate optimal max_length based on input text length and target language.
    Some languages need more tokens than others.
    """
    base_length = len(text.split()) * 2  # Rough estimate

    # Language-specific multipliers (some languages need more tokens)
    multipliers = {
        "eng": 1.0,
        "ach": 1.2,
        "lgg": 1.1,
        "lug": 1.3,
        "nyn": 1.1,
        "teo": 1.2,
    }

    multiplier = multipliers.get(target_language, 1.2)
    optimal_length = int(base_length * multiplier)

    # Ensure reasonable bounds
    return max(50, min(optimal_length, 512))


def translate_batch(
    texts: List[str], source_language: str, target_language: str
) -> List[str]:
    """
    Translate multiple texts in a single batch for better GPU utilization.
    """
    logger.info(
        f"Translating batch of {len(texts)} texts from {source_language} to {target_language}"
    )
    logger.debug(f"Texts: {texts}")

    if not texts:
        return []

    _language_codes = {
        "eng": 256047,
        "ach": 256111,
        "lgg": 256008,
        "lug": 256110,
        "nyn": 256002,
        "teo": 256006,
    }

    # Tokenize all texts at once
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    # Set source language token for all sequences
    inputs["input_ids"][:, 0] = _language_codes[source_language]

    # Calculate optimal max length based on the actual longest text in the batch
    # (previous code built a synthetic repeated-number string which often resulted
    #  in a small, fixed max_length of 50 and caused truncation of outputs).
    longest_text = max(texts, key=lambda t: len(t.split()))
    max_length = get_optimal_max_length(longest_text, target_language)
    logger.info(f"Using max_length={max_length} for batch translation")

    with torch.no_grad():  # Disable gradients for inference
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=_language_codes[target_language],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            do_sample=False,  # deterministic output
            pad_token_id=tokenizer.pad_token_id,
        )

    results = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return results


def translate(text: str, source_language: str, target_language: str) -> str:
    """
    Optimized single text translation.
    """
    return translate_batch([text], source_language, target_language)[0]


def smart_chunk_by_sentences(text: str, max_chunk_size: int = 100) -> List[str]:
    """
    Improved sentence-based chunking that respects context and optimal batch sizes.
    """
    # Enhanced sentence splitting with more delimiters
    sentence_endings = r"([.!?।।।])\s*"
    sentences = re.split(sentence_endings, text)

    # Recombine sentences with their punctuation
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + (
                sentences[i + 1] if sentences[i + 1] in ".!?।।।" else ""
            )
            if sentence.strip():
                combined_sentences.append(sentence.strip())

    # If no sentences found, fall back to word chunking
    if not combined_sentences:
        return chunk_by_words(text, max_chunk_size)

    # Group sentences into optimal chunks
    chunks = []
    current_chunk = ""

    for sentence in combined_sentences:
        # If adding this sentence would exceed max_chunk_size
        if (
            current_chunk
            and len(current_chunk.split()) + len(sentence.split()) > max_chunk_size
        ):
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    logger.info(f"Text chunked into {len(chunks)} parts using sentence-based chunking.")
    logger.debug(f"Chunks: {chunks}")

    return chunks


def chunk_by_words(text: str, chunk_size: int = 80) -> List[str]:
    """
    Optimized word-based chunking with better size management.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))

    logger.info(f"Text chunked into {len(chunks)} parts using word-based chunking.")
    logger.debug(f"Chunks: {chunks}")

    return chunks


def contains_delimiters(text: str) -> bool:
    """
    Enhanced delimiter detection.
    """
    pattern = r"[.?!।।।]"
    return bool(re.search(pattern, text))


def process_and_translate_text(
    text: str, source_language: str, target_language: str
) -> str:
    """
    Optimized text processing and translation with batch processing.
    """
    start_time = time.time()
    if not text.strip():
        return ""

    # For short text, translate directly
    if len(text.split()) <= 30:
        return translate(text, source_language, target_language)

    # For longer text, use intelligent chunking
    if contains_delimiters(text):
        chunks = smart_chunk_by_sentences(text, max_chunk_size=80)
    else:
        chunks = chunk_by_words(text, chunk_size=80)

    # Batch process chunks for better GPU utilization
    batch_size = 4  # Process 4 chunks at once
    all_translations = []

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_translations = translate_batch(
            batch_chunks, source_language, target_language
        )
        all_translations.extend(batch_translations)

    end_time = time.time()
    logger.info(f"Translation completed in {end_time - start_time:.2f} seconds")

    return " ".join(all_translations)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    test_text = (
        "Randy Bruce Traywick (born May 4, 1959), known professionally as Randy Travis, "
        "is an American country and gospel music singer and songwriter, as well as a film "
        "and television actor. Active since 1979, he has recorded over 20 studio albums "
        "and charted over 50 singles on the Billboard Hot Country Songs charts, including "
        "sixteen that reached the number-one position. Travis's commercial success began "
        "in the mid-1980s with the release of his album Storms of Life, which was certified "
        "triple-platinum by the Recording Industry Association of America. He followed up "
        "his major-label debut with a string of platinum and multi-platinum albums, but his "
        "commercial success declined throughout the 1990s. In 1997, he left Warner Bros. "
        "Records for DreamWorks Records; he signed to Word Records for a series of gospel "
        "albums beginning in 2000 before transferring back to Warner at the end of the 21st "
        "century's first decade. His musical accolades include seven Grammy Awards, eleven "
        "ACM Awards, eight Dove Awards, a star on the Hollywood Walk of Fame, and a 2016 "
        "induction into the Country Music Hall of Fame. Major songs of his include 'On the "
        "Other Hand', 'Forever and Ever, Amen', 'I Told You So', 'Hard Rock Bottom of Your "
        "Heart', and 'Three Wooden Crosses'."
    )
    test_text = "Randy Bruce Traywick (born May 4, 1959), known professionally as Randy Travis, is an American country and gospel music singer and songwriter, as well as a film and television actor. Active since 1979, he has recorded over 20 studio albums and charted over 50 singles on the Billboard Hot Country Songs charts, including sixteen that reached the number-one position."
    wiki_text = read_text_file("./wikimedia-samples/wikimedia_text.txt")
    logger.info("Starting translation...")
    logger.debug(f"Original Text: {wiki_text}")
    translated = process_and_translate_text(wiki_text, "eng", "lug")
    logger.debug(f"Translated Text: {translated}")
    write_text_file("./wikimedia-samples/translated_output.txt", translated)
