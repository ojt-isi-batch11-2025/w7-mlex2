#!/bin/bash
PYTHON_VERSION="3.10.13"
# if .venv does not exist, then create it
if [ ! -d ".venv" ]; then
    uv venv .venv --python=${PYTHON_VERSION}
fi
# if which python does not return the path to the virtual environment's python executable, then activate the virtual environment
if [ "$(which python)" != "$(readlink -f .venv/Scripts/python)" ]; then
    source .venv/Scripts/activate
fi
uv pip install -r .requirements.txt