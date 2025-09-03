#!/bin/bash

CURRENT_DIR=$(dirname "$0")

if ! command -v "uv" &> /dev/null; then
    echo "Please install uv from https://docs.astral.sh/uv/getting-started/installation/#installation-methods and try again"
    exit 1
fi

if [ -d "${CURRENT_DIR}/models" ]; then
    echo "Directory "${CURRENT_DIR}/models" already exists. Removing directory."
    rm -r "${CURRENT_DIR}/models"
fi

mkdir -p "${CURRENT_DIR}/models"
cd "${CURRENT_DIR}/models"

echo "Downloading landmark.pth"
uv tool run --isolated gdown https://drive.google.com/file/d/1NckKw7elDjQTllRxttO87WY7cnQwdMqz/view --fuzzy
mv checkpoint_landmark_191116.pth.tar landmark.pth

echo "Downloading lbpcascade_animeface.xml"
wget https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml
mv lbpcascade_animeface.xml lbpcascade.xml

echo "Downloading segment.safetensors"
uv tool run --isolated --from huggingface-hub hf download --local-dir . skytnt/anime-seg model.safetensors
mv model.safetensors segment.safetensors
