import logging
import os
import time

import requests
import runpod
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
# Set RunPod API Key
runpod.api_key = os.getenv("RUNPOD_API_KEY")

endpoint = runpod.Endpoint(RUNPOD_ENDPOINT_ID)
request_response = {}

text = (
    "matuggakyenkugamba kiri kimu nti gavumenti erina okuteekawo enkola kiseera smesaabantuakabyata "
    "kale muko eteekewo n'etteeka nti omwana okufuna olubuto nga tannaweza myaka kumi na munaana alimufunisizza "
    "kalabba naye musango naye bw'ovutukogamba mbu ba nabba ku bampisa boyononakabalaba katiwewakawakanyizibwa "
    "bateekewo eteeka nti okukwata oba sibwa emyaka abiri singa ofunizizaomwana olubuto nga tannaba kwetuuka"
)


try:
    request_response = endpoint.run_sync(
        {
            "input": {
                "task": "asr_summarise",
                "source_language": "lug",
                "target_language": "eng",
                "text": text,
            }
        },
        timeout=600,  # Timeout in seconds.
    )

    # Log the request for debugging purposes
    logging.info(f"Request response: {request_response}")

except TimeoutError:
    # Handle timeout error and return a meaningful message to the user
    logging.error("Job timed out.")
    raise TimeoutError(
        "The asr text suummarisation job timed out. Please try again later.",
    )

print(request_response)
