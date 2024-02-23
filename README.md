[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

# asr-evaluate

## Requirements

- Python >= 3.10

## Installation

- Install dependencies with `pip install -r requirements_cpu.txt` or with `pip install -r requirements_gpu.txt` according to your system.

## Usage

Right now, only Whisper can be used for evaluation.

### Evaluating datasets

- Use evaluate_dataset.py to evaluate either the CommonVoice or the Fleurs dataset.

### Evaluating larger files and transcripts

- Work in progress, try out vtt_cleaning.py or wer_calculation.py
