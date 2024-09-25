import getpass
import os

runpod_api_key = getpass.getpass("Enter your RUNPOD_API_KEY: ")
os.environ["RUNPOD_API_KEY"] = runpod_api_key
print(f"Runpod API KEY: {os.getenv('RUNPOD_API_KEY')}")
