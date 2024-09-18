#!/usr/bin/env bash

echo "Creating virtual environment"
python3.10 -m venv mfix-env
echo "Activating virtual environment"

source $PWD/mfix-env/bin/activate

$PWD/mfix-env/bin/pip install --upgrade pip setuptools

$PWD/mfix-env/bin/pip install torch torchvision torchaudio
# torch 2.0.0 cu117

$PWD/mfix-env/bin/pip install pytorch-lightning==2.2.4
$PWD/mfix-env/bin/pip install -r requirements.txt
