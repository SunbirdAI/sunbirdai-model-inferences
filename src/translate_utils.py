import torch
import transformers

tokenizer = transformers.NllbTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-1.3B"
)
model = transformers.M2M100ForConditionalGeneration.from_pretrained(
    "jq/nllb-1.3B-many-to-many-pronouncorrection-charaug"
)
import re

try:
    device = torch.device("cuda")
except Exception:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_response(response):
    return {"data": response}


def translate(text, source_language, target_language):
    _language_codes = {
        "eng": 256047,
        "ach": 256111,
        "lgg": 256008,
        "lug": 256110,
        "nyn": 256002,
        "teo": 256006,
    }

    inputs = tokenizer(text, return_tensors="pt").to(device)
    inputs["input_ids"][0][0] = _language_codes[source_language]
    translated_tokens = model.to(device).generate(
        **inputs,
        forced_bos_token_id=_language_codes[target_language],
        max_length=100,
        num_beams=5,
    )

    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return result


def chunk_text(text, chunk_size=20):
    """
    Split the text into chunks of specified size (default: 50 words).
    """
    words = text.split()
    chunks = [words[i : i + chunk_size] for i in range(0, len(words), chunk_size)]
    return [" ".join(chunk) for chunk in chunks]


def split_text_with_delimiters(text):
    """
    Splits the input text at each occurrence of '.', '?', or '!',
    while maintaining these delimiters in the resulting split portions.

    Args:
      text (str): The input text to be split.

    Returns:
      list: A list of strings where each string is a segment of the original
            text ending with one of the specified delimiters.

    Example:
      >>> text = "Hello! How are you doing? I hope you're doing well. Have a great day!"
      >>> split_text_with_delimiters(text)
      ['Hello! How are you doing?', "I hope you're doing well.", 'Have a great day!']
    """
    # Define the regular expression pattern to match '.', '?', or '!', capturing them
    pattern = r"([.?!])"

    # Split the text based on the pattern, including the delimiters
    split_result = re.split(pattern, text)

    # Combine each delimiter with the preceding text part
    combined_result = [
        split_result[i] + split_result[i + 1]
        for i in range(0, len(split_result) - 1, 2)
    ]

    # If the text does not end with a delimiter, append the last part
    if len(split_result) % 2 != 0 and split_result[-1].strip():
        combined_result.append(split_result[-1])

    # Remove any empty strings from the result
    combined_result = [part for part in combined_result if part.strip()]

    return combined_result


def contains_delimiters(text):
    """
    Checks if the input text contains any of the delimiters: '.', or '?' or '!'.

    Args:
        text (str): The input text to be checked.

    Returns:
        bool: True if any of the delimiters are found, False otherwise.

    Example:
        >>> contains_delimiters("Hello! How are you?")
        True
        >>> contains_delimiters("Hello How are you")
        False
    """
    # Define the regular expression pattern to match '.', or '?'
    pattern = r"[.?!]"
    # Search for the pattern in the text
    return bool(re.search(pattern, text))


def translate_text_with_delimiters(text, source_language, target_language):
    """
    Translates the input text from the source language to the target language.
    If the text contains delimiters ('.', '?'), splits the text first and translates each part separately.

    Args:
        text (str): The input text to be translated.
        source_language (str): The source language code.
        target_language (str): The target language code.

    Returns:
        list: A list of translated sentences.

    Example:
        >>> text = "Hello! How are you?"
        >>> translate_text_with_delimiters(text, "eng", "lug")
        ['Mwasuze mutya?']
    """
    if contains_delimiters(text):
        sentences = split_text_with_delimiters(text)
    else:
        sentences = chunk_text(text, chunk_size=20)

    translated_sentences = [
        translate(sentence.strip(), source_language, target_language)
        for sentence in sentences
    ]

    return translated_sentences


def process_and_translate_text(text, source_language, target_language):
    """
    Processes the input text by checking for delimiters, splitting if necessary,
    translating each part, and then joining the translated sentences.

    Args:
        text (str): The input text to be processed and translated.
        source_language (str): The source language code.
        target_language (str): The target language code.

    Returns:
        str: The translated text.

    Example:
        >>> text = "Hello! How are you?"
        >>> process_and_translate_text(text, "eng", "lug")
        'Mwasuze mutya?'
    """
    translated_sentences = translate_text_with_delimiters(
        text, source_language, target_language
    )
    translated_text = " ".join(translated_sentences)
    return translated_text


if __name__ == "__main__":
    # Example usage
    text = "Hello! How are you doing? I hope you're doing well. Have a great day!"
    split_result = split_text_with_delimiters(text)
    print(split_result)

    text_with_delimiters = "Hello! How are you?"
    text_without_delimiters = "Hello How are you"
    print(contains_delimiters(text_with_delimiters))  # Output: True
    print(contains_delimiters(text_without_delimiters))  # Output: False

    text = "Hello! How are you doing? I hope you're doing well. Have a great day!"
    source_language = "eng"
    target_language = "lug"
    translated_text = process_and_translate_text(text, source_language, target_language)
    print(translated_text)
