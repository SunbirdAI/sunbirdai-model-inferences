import torch
import transformers

tokenizer = transformers.NllbTokenizer.from_pretrained(
    "./model-weights/nllb-1.3B-asr-summarisation"
)
model = transformers.M2M100ForConditionalGeneration.from_pretrained(
    "./model-weights/nllb-1.3B-asr-summarisation"
)

try:
    device = torch.device("cuda")
except Exception:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        max_length=1024,
        num_beams=20,
    )

    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return result


def chunk_text(text, chunk_size=50):
    """
    Split the text into chunks of specified size (default: 50 words).
    """
    words = text.split()
    chunks = [words[i : i + chunk_size] for i in range(0, len(words), chunk_size)]
    return [" ".join(chunk) for chunk in chunks]


def correct_text_chunks(text_chunks, source_language="eng"):
    """
    Corrects a list of text chunks using a translation function.

    This function iterates over a list of text chunks, applies a translation function to each chunk to correct it,
    and returns a list of corrected text chunks. The translation function is assumed to correct the text by translating
    it from the specified source language to the same language.

    Args:
        text_chunks (list of str): A list of text chunks to be corrected. Each chunk is expected to be a string.
        source_language (str): The language code of the source text. Defaults to "eng".

    Returns:
        list of str: A list of corrected text chunks.

    Example:
        text_chunks = ["This is an exmaple chunk.", "Another chnk for correction."]
        corrected_chunks = correct_text_chunks(text_chunks, source_language="eng")
        print(corrected_chunks)
        # Output: ["This is an example chunk.", "Another chunk for correction."]
    """
    corrected_text_chunks = []
    for chunk in text_chunks:
        corrected_chunk = translate(chunk.strip(), source_language, source_language)
        corrected_text_chunks.append(corrected_chunk)
    return corrected_text_chunks


def join_corrected_text_chunks(corrected_text_chunks):
    """
    Joins a list of corrected text chunks into a single corrected text string.

    This function takes a list of corrected text chunks and concatenates them into a single string, separated by spaces.
    It ensures that the final output is a cohesive and readable text.

    Args:
        corrected_text_chunks (list of str): A list of corrected text chunks. Each chunk is expected to be a string.

    Returns:
        str: A single string containing the concatenated corrected text chunks.

    Example:
        corrected_text_chunks = ["This is an example chunk.", "Another chunk for correction."]
        final_corrected_text = join_corrected_text_chunks(corrected_text_chunks)
        print(final_corrected_text)
        # Output: "This is an example chunk. Another chunk for correction."
    """
    final_corrected_text = " ".join(corrected_text_chunks)
    return final_corrected_text


def process_and_correct_text(text, chunk_size=50, source_language="eng"):
    """
    Processes and corrects text by chunking it, correcting each chunk, and returning the corrected text.

    This function takes an input text, splits it into chunks of a specified size, corrects each chunk using a translation
    function, and then concatenates the corrected chunks into a single corrected text string.

    Args:
        text (str): The input text to be processed and corrected.
        chunk_size (int): The number of words per chunk. Defaults to 50.
        source_language (str): The language code of the source text. Defaults to "eng".

    Returns:
        str: The final corrected text after chunking and correcting each chunk.

    Example:
        text = "This is a sample text that needs to be chunked and corrected."
        corrected_text = process_and_correct_text(text, chunk_size=5, source_language="eng")
        print(corrected_text)
        # Output: "This is a sample text that needs to be chunked and corrected."
    """
    # Split text into chunks
    text_chunks = chunk_text(text, chunk_size)

    # Correct the chunks
    corrected_text_chunks = correct_text_chunks(text_chunks, source_language)

    # Join the corrected chunks into final corrected text
    final_corrected_text = join_corrected_text_chunks(corrected_text_chunks)

    return final_corrected_text


if __name__ == "__main__":
    text = (
        "welcome to she center the female podcast i am rest consume and my mission is "
        "to create a safe space for women to share their stories and triumphs she sent "
        "is more than just a podcast is a in a car with brillian mind l digital platform "
        "that serves women in uganda and the rest of east africa empowering them to thrive "
        "in all aspects of their lives through this podcast in a to bring healing and car e "
        "lessons by shockesingthe power of story telling the episodes who future women from "
        "divest backgrounds sharing their personal expense and a and istafrika i biliv in da "
        "global sister so weda yu listening from uganda east africa or anywhere in de world "
        "join us on this journey of self discovery growth and empowerment let's amplify the "
        "voices of women break barriers challenge stereotypes and create a network of support "
        "and empawrmenttogether we can bring about positive change and make this world a better "
        "place for all this is she center the female podcast where women's stories mutter"
    )

    corrected_text = process_and_correct_text(
        text, chunk_size=50, source_language="eng"
    )
