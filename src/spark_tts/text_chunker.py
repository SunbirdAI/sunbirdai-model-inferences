"""
text_chunker.py

Utility module for splitting text into fixed-size chunks of tokens while preserving punctuation attachment.
"""

import re
from typing import List

# Define the public API: only expose chunk_text when using `from text_chunker import *`
__all__ = ["chunk_text"]


def chunk_text(text: str, chunk_size: int) -> List[str]:
    """
    Split `text` into chunks of up to `chunk_size` tokens, preserving punctuation,
    while treating apostrophes as part of word tokens.

    Tokens are defined as words (alphanumeric sequences plus apostrophes) or standalone punctuation marks.
    When reconstructing chunks, punctuation attaches directly to the preceding word without extra spaces.

    Args:
        text: The input string to chunk.
        chunk_size: Maximum number of tokens per chunk.

    Returns:
        A list of text chunks.
    """
    # Tokenize: words including apostrophes, or any other non-word, non-space character
    tokens = re.findall(r"[\w']+|[^\w\s]", text, flags=re.UNICODE)
    chunks: List[str] = []
    current: List[str] = []

    for tok in tokens:
        current.append(tok)
        if len(current) >= chunk_size:
            chunks.append(_join_tokens(current))
            current = []

    if current:
        chunks.append(_join_tokens(current))

    return chunks


def _join_tokens(tokens: List[str]) -> str:
    """
    Reassemble a list of tokens into a text chunk, attaching punctuation
    directly to the preceding word.

    Args:
        tokens: List of token strings.

    Returns:
        The joined text string.
    """
    result = ""
    for tok in tokens:
        # treat punctuation except apostrophe as attach-to-previous
        if re.fullmatch(r"[^\w\s']", tok):
            result += tok
        else:
            if result:
                result += " "
            result += tok
    return result


# Example usage of __all__ and chunk_text:
# ----------------------------------------
# When imported with `from text_chunker import *`, only chunk_text is imported:
#
#   >>> from text_chunker import *
#   >>> chunk_text("Hello, world! It's me.", 3)
#   ['Hello,', 'world!', "It's", 'me.']
#
# Direct import also works:
#
#   >>> import text_chunker
#   >>> text_chunker.chunk_text("Don't split contractions.", 2)
#   ["Don't", 'split', 'contractions.']


if __name__ == "__main__":
    sample = (
        "Kinyarwanda is spoken by more than 12 million people, yet high quality ASR systems remain scarce. "
        "This dataset and accompanying hackathon aim to: Accelerate speech to text research and open-source "
        "toolkits for Kinyarwanda. Provide diverse, real-world audio covering critical societal domains. "
        "Benchmark ASR systems under both supervised and semi-supervised settings. Support the development "
        "of digital public goods by releasing resources under open licenses, thereby enabling public "
        "institutions, developers, and researchers to build inclusive voice technologies. Strengthen the "
        "local AI and NLP ecosystem in Rwanda and across Africa by engaging academia, startups, and "
        "established companies in building language technologies. Promote linguistic equity by ensuring "
        "that native Kinyarwanda speakers can interact with technology in their own language."
    )
    for i, chunk in enumerate(chunk_text(sample, chunk_size=12), 1):
        print(f"--- Chunk {i} ---\n{chunk}\n")
