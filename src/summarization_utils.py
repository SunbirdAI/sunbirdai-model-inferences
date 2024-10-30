import logging

import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

try:
    device = torch.device("cuda")
except Exception:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)


def summarize_text(input_text):
    """
    Summarize the given input text using a pre-trained language model.

    This function uses a fine-tuned model to generate an anonymized summary of the provided text.
    The model is loaded with specific configurations including 4-bit quantization to optimize memory usage.

    Parameters:
    input_text (str): The input text to be summarized.

    Returns:
    str: The anonymized summary of the input text.
    """
    try:
        logging.info(f"device: {device}")
        max_seq_length = 2048  # Define the maximum sequence length
        dtype = None  # Auto-detect the appropriate data type
        load_in_4bit = True  # Enable 4-bit quantization for memory optimization

        # Load the pre-trained model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="akera/sunflower-llama3-finetuned-20240424",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # Define the instruction for the model
        instruction = "Anonymised summary"

        # Format the prompt
        instruction_prompt = """{}
    
        ### Input
        {}
        
        ### Output
        """

        # Tokenize the input text
        inputs = tokenizer(
            [
                instruction_prompt.format(
                    instruction,
                    input_text.lower() if "summary" in instruction else input_text,
                )
            ],
            return_tensors="pt",
        ).to(device)

        # Initialize the text streamer
        text_streamer = TextStreamer(tokenizer)

        # Generate the summary using the model
        result = model.generate(
            **inputs, streamer=text_streamer, max_new_tokens=150, repetition_penalty=1.1
        )

        # Decode the generated tokens to get the output text
        output_text = tokenizer.decode(result[0], skip_special_tokens=True)

        # Extract the part of the text after '### Output'
        if "### Output" in output_text:
            output_text = output_text.split("### Output")[-1].strip()
    except Exception as e:
        logging.info(str(e))
        output_text = str(e)

    return output_text


if __name__ == "__main__":
    # Example usage
    input_text = (
        "Newaakubadde nga teyakyatula naye kyali kimanyiddwa nga bwe "
        "yeegombanga ennyo okubeera ne Bokisa. Buli lwa Sande bombi baateranga "
        "okulabibwa nga balya omuddo mu kataletale akaali keesudde ennimiro "
        "y'ebibala, kyokka nga ku bombi tewali anyega munne."
    )

    summary = summarize_text(input_text)
    print(summary)
