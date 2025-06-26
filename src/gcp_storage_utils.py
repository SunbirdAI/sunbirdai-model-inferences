import datetime
import os

from google.cloud import storage

from transcribe_utils import decode_gcp_credentials


def upload_audio_file_to_gcs(
    local_file_path: str,
    bucket_name: str,
    destination_blob_name: str,
    expiration_minutes: int = 30,
) -> str:
    """
    Upload a local audio file to Google Cloud Storage and return a presigned URL.

    Args:
        local_file_path: Path to the local audio file to upload.
        bucket_name: Name of the target GCS bucket.
        destination_blob_name: Blob name to assign in the bucket (e.g., 'folder/file.mp3').
        expiration_minutes: URL expiration time in minutes (default: 30).

    Returns:
        A signed URL (str) granting temporary read access to the uploaded file.

    Raises:
        FileNotFoundError: If the local file does not exist.
        google.cloud.exceptions.GoogleCloudError: On upload or signing failure.
    """
    # Ensure the file exists locally
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"File not found: {local_file_path}")

    # Obtain service account credentials
    # decode_gcp_credentials() will write credentials.json if needed
    if os.path.exists("credentials.json"):
        credentials_path = "credentials.json"
    else:
        credentials_path = decode_gcp_credentials()

    # Initialize the GCS client
    client = storage.Client.from_service_account_json(credentials_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Upload the file
    blob.upload_from_filename(local_file_path)

    # Generate a signed URL (v4) valid for the specified duration
    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=expiration_minutes),
        method="GET",
    )

    return signed_url
