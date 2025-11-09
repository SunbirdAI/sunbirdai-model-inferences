"""
text_chunker.py

Utility module for splitting text into fixed-size chunks of tokens while preserving punctuation attachment.
"""

import re
from typing import List

# Define the public API: only expose chunk_text when using `from text_chunker import *`
__all__ = ["chunk_text"]


def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Split text into chunks based on sentence boundaries (periods, exclamation marks, question marks).
    
    This approach is ideal for TTS as it:
    - Preserves natural sentence flow and intonation
    - Avoids cutting off mid-sentence
    - Groups multiple sentences together when they're short
    
    Args:
        text: The input string to chunk.
        max_chunk_size: Maximum character length per chunk (soft limit). 
                       Sentences won't be split even if they exceed this.
    
    Returns:
        A list of text chunks, each containing one or more complete sentences.
    """
    # Split on sentence-ending punctuation while preserving the punctuation
    # This regex splits on . ! ? followed by whitespace or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed max_chunk_size and we already have content,
        # save the current chunk and start a new one
        if current_chunk and (current_length + sentence_length + 1) > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_length + 1  # +1 for space
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def chunk_text_simple(text: str) -> List[str]:
    """
    Split text into individual sentences.
    
    Use this for maximum control in TTS - one sentence per chunk.
    
    Args:
        text: The input string to chunk.
    
    Returns:
        A list of sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_with_count(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """
    Split text into chunks containing a specific number of sentences.
    
    Args:
        text: The input string to chunk.
        sentences_per_chunk: Number of sentences to include in each chunk.
    
    Returns:
        A list of text chunks.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks: List[str] = []
    
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    
    return chunks


# Example usage and testing
if __name__ == "__main__":
    sample_text = """Hello, I'm Prosi Nafula. I am a nurse who takes care of many people who have cancer and who have questions about their illness and what to expect. There are many types of cancer. The type of cancer you have is named after the place where it started. For example, if cancer starts in the breast then it is called breast cancer."""
    
    print("=== Default chunking (by sentence boundaries, max 200 chars) ===")
    chunks = chunk_text(sample_text, max_chunk_size=200)
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk)
    
    print("\n\n=== Simple chunking (one sentence per chunk) ===")
    chunks = chunk_text_simple(sample_text)
    for i, chunk in enumerate(chunks, 1):
        print(f"\nSentence {i}:")
        print(chunk)
    
    print("\n\n=== Fixed sentence count (2 sentences per chunk) ===")
    chunks = chunk_text_with_count(sample_text, sentences_per_chunk=2)
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)