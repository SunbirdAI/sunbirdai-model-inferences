import torch
import transformers

tokenizer = transformers.NllbTokenizer.from_pretrained("jq/nllb-1.3B-asr-summarisation")
model = transformers.M2M100ForConditionalGeneration.from_pretrained(
    "jq/nllb-1.3B-asr-summarisation"
)


def translate(text, source_language, target_language):
    _language_codes = {
        "eng": 256047,
        "ach": 256111,
        "lgg": 256008,
        "lug": 256110,
        "nyn": 256002,
        "teo": 256006,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(text, return_tensors="pt").to(device)
    inputs["input_ids"][0][0] = _language_codes[source_language]
    translated_tokens = model.to(device).generate(
        **inputs,
        forced_bos_token_id=_language_codes[target_language],
        max_length=250,
        num_beams=20,
    )

    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return result
