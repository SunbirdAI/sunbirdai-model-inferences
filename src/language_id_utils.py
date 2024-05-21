from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("yigagilbert/salt_language_ID")
model = AutoModelForSeq2SeqLM.from_pretrained("yigagilbert/salt_language_ID")
