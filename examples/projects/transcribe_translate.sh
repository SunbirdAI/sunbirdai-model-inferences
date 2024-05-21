#!/bin/bash

# Set the directory containing language subdirectories
directory=$1
# Set the path to the output CSV file
output_file=$2
# Set the authentication token for accessing the API
auth_token=$3

# Execute the Python program
python transcribe_translate.py "$directory" "$output_file" --auth_token "$auth_token"
