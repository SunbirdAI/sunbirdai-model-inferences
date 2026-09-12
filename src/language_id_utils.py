import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSIFICATION_MODEL_NAME = "Sunbird/sunflower_language_classification_v2"

# use_fast=False is mandatory: the fast-tokenizer conversion drops this
# model's SentencePiece byte_fallback and reintroduces <unk> on scripts like
# Amharic (0.47% vs 0.00% with the slow tokenizer), which is the exact
# short-text regression v2 was retrained to fix.
classification_tokenizer = AutoTokenizer.from_pretrained(
    CLASSIFICATION_MODEL_NAME,
    use_fast=False,
)
classification_model = AutoModelForSequenceClassification.from_pretrained(
    CLASSIFICATION_MODEL_NAME
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace, exactly as v2 was trained on."""
    return " ".join(text.lower().split())


def predict(text, device):
    """
    Perform inference on the input text and return the probabilities for each label.

    This function tokenizes the input text, performs inference using a pre-trained model,
    and calculates the probability for each label using softmax. The result is a dictionary
    mapping each label to its corresponding probability.

    Args:
        text (str): The input text to perform inference on.
        device (torch.device): The device to run the model on.

    Returns:
        dict: A dictionary where keys are the labels (e.g., "eng", "lug", "ach", "teo", "lgg", "nyn")
              and values are the corresponding probabilities as floats.

    Example:
        >>> result = predict("example text", device)
        >>> print(result)
        {'eng': 0.2, 'lug': 0.1, 'ach': 0.4, 'teo': 0.1, 'lgg': 0.15, 'nyn': 0.05}

    Note:
        - This function assumes that the `classification_tokenizer` and `classification_model` are already defined
          and properly set up in the global scope.
        - The `torch` library is required for tensor operations.
        - The `device` is automatically handled within the function.
    """
    classification_model.to(device)

    # Both callers (auto_detect_language and language_classify) depend on
    # this matching training's normalization exactly, or short-text accuracy
    # silently degrades.
    inputs = classification_tokenizer(
        _normalize_text(text), return_tensors="pt", truncation=True, padding=True
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = classification_model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Map labels to their respective probabilities using classification_model.config.id2label
    label_mapping = classification_model.config.id2label

    # Get all probabilities with their labels
    all_predictions = {
        label_mapping[i]: float(probability)
        for i, probability in enumerate(probabilities)
    }

    # Sort predictions by probability in descending order and take the top 6
    sorted_predictions = sorted(all_predictions.items(), key=lambda item: item[1], reverse=True)
    top_6_predictions = dict(sorted_predictions[:6])

    return top_6_predictions
