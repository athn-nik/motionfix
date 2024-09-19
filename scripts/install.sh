#!/usr/bin/env bash

echo "Creating virtual environment"
python3.10 -m venv mfix-env
echo "Activating virtual environment"

source $PWD/mfix-env/bin/activate

$PWD/mfix-env/bin/pip install --upgrade pip setuptools

$PWD/mfix-env/bin/pip install "torch==2.1.2" "torchvision==0.16.2" --index-url https://download.pytorch.org/whl/cu118

$PWD/mfix-env/bin/pip install "pytorch-lightning==2.2.4"
$PWD/mfix-env/bin/pip install -r requirements.txt
