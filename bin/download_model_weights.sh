#!/bin/bash

set -e

# Directory to store model weights
TARGET_DIR="model-weights"

# Ensure Git LFS is installed and initialized
git lfs install

# Array of model repositories
REPOS=(
  "https://huggingface.co/Sunbird/asr-mms-salt"
  "https://huggingface.co/Sunbird/translate-nllb-1.3b-salt"
  "https://huggingface.co/facebook/nllb-200-distilled-1.3B"
  "https://huggingface.co/jq/nllb-1.3B-many-to-many-pronouncorrection-charaug"
  "https://huggingface.co/jq/whisper-large-v2-salt-plus-xog-myx-kin-swa-sample-packing"
  "https://huggingface.co/yigagilbert/salt_language_ID"
  "https://huggingface.co/yigagilbert/salt_language_Classification"
  "https://huggingface.co/jq/nllb-1.3B-asr-summarisation"
  "https://huggingface.co/jq/whisper-large-v2-multilingual-prompts-corrected"
)

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Process each repository
for REPO in "${REPOS[@]}"; do
  REPO_NAME=$(basename "$REPO")
  REPO_PATH="$TARGET_DIR/$REPO_NAME"

  if [ -d "$REPO_PATH/.git" ]; then
    # If the repository already exists, update it
    echo "Repository $REPO_NAME already exists. Pulling latest changes..."
    cd "$REPO_PATH" || exit
    git reset --hard  # Ensure a clean working directory
    git pull origin main  # Pull the latest changes from the main branch
    git lfs pull  # Pull large files if needed
    cd - || exit
  else
    # Clone the repository if it doesn't exist
    echo "Cloning repository $REPO_NAME..."
    git clone "$REPO" "$REPO_PATH"
    cd "$REPO_PATH" || exit
    git lfs pull  # Pull large files
    cd - || exit
  fi
done

echo "All repositories have been updated or cloned with large files downloaded."
