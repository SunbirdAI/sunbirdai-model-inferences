""" Example handler file. """

import runpod
import os

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory)

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
from translate_utils import make_response, translate



def handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job["input"]

    source_language = job_input.get("source_language")
    target_language = job_input.get("target_language")
    text_to_translate = job_input.get("text")

    translated_text = translate(text_to_translate, source_language, target_language)

    resp = {"text": text_to_translate, "translated_text": translated_text}

    return make_response(response=resp)


runpod.serverless.start({"handler": handler})