# asr-evaluate

[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

## Requirements

- Python >= 3.10

## Installation

Install Pytorch [according to your system,](https://pytorch.org/get-started/locally/) e.g. for Linux with only CPU support:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Install the rest of the dependencies:

```shell
pip install -r requirements.txt
```

Create the configuration file.

```shell
cp config.example.toml config.toml
```

## Usage

Right now, only Whisper can be used for evaluation.

### Evaluating datasets

Use evaluate_dataset.py to evaluate either the Fleurs or the CommonVoice dataset.

```bash
python evaluate_dataset.py fleurs
```

Use the length option to reduce the number of data points.

```bash
python evaluate_dataset.py cv length=100
```

### Evaluating larger files and transcripts

Work in progress, try out vtt_cleaning.py or wer_calculation.py
