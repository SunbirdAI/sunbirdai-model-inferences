# Audio Transcription and Translation Pipeline

This pipeline script transcribes and translates audio files in a directory and its subdirectories using an AI API. It produces a CSV file containing the transcription and translation results.

## Prerequisites

Before using the pipeline script, ensure you have the following:

- Python installed on your system (version 3.8 or higher)
- Required Python libraries installed:
  - pandas
  - requests

## Installation

1. Clone this repository to your local machine:

```sh
git clone https://github.com/SunbirdAI/sunbird-nllb-translate-inference-server.git
```

2. Navigate to the cloned repository:

```sh
cd sunbird-nllb-translate-inference-server
```

3. Install the required Python libraries:

Make sure you are in the root directory of the repository

```sh
pip install -r requirements-dev.txt
pip install -r builder/requirements.txt
```

## Usage

To transcribe and translate audio files in a directory, follow these steps:

1. **Prepare Audio Files:**

   Place your audio files in a directory structure where each subdirectory represents a different language. For example:

   ```
   ├── trac_fm
   │   └── ach
   │       ├── MEGA 12.1.mp3
   │       └── MEGA 12.2.mp3
   │   └── lug
   │       ├── audio1.mp3
   │       └── audio2.mp3
   ```

2. **Run the Pipeline Script:**

   Execute the pipeline script from the command line, providing the directory containing the audio files, the path to the output CSV file, and your authentication token:

   ```sh
   ./transcribe_translate.sh trac_fm output.csv your_auth_token_here
   ```

   Replace `trac_fm` with the path to your directory containing language subdirectories, `output.csv` with the desired name for the output CSV file, and `your_auth_token_here` with your actual authentication token.

3. **View Results:**

   After the script finishes execution, you will find the output CSV file (`output.csv`) containing the transcription and translation results.

## Example

Suppose you have a directory named `trac_fm` containing audio files in different languages. You want to transcribe and translate these files to English. Here's how you would use the pipeline script:

1. Navigate to the directory containing the pipeline script:

   ```sh
   cd sunbird-nllb-translate-inference-server/examples/projects
   ```

2. Run the pipeline script:

   ```sh
   chmod u+x transcribe_translate.sh
   AUTH_TOKEN=your_auth_token_here
   ./transcribe_translate.sh trac_fm trac_fm_output.csv $AUTH_TOKEN
   ```

   This command will transcribe and translate the audio files in the `trac_fm` directory and save the results to `trac_fm_output.csv`.

3. View the results:

   Open the `trac_fm_output.csv` file to view the transcription and translation results.

Also trying with sample data of `backup uganda` Acholi Audios.

```sh
./transcribe_translate.sh backup_uganda backup_uganda_output.csv $AUTH_TOKEN
```
