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
pip install git+https://github.com/m-bain/whisperx.git --upgrade
```

Create the configuration file.

```shell
cp config.example.toml config.toml
```

## Usage

Right now, only Whisper can be used for evaluation.

### Evaluating datasets

Login with huggingface_hub token, requires a token generated from https://huggingface.co/settings/tokens .

```shell
huggingface-cli login
```

Use evaluate_dataset.py to evaluate either the Fleurs or the CommonVoice dataset.

```shell
python evaluate_dataset.py fleurs
```

Use the length option to reduce the number of data points.

```shell
python evaluate_dataset.py cv --length=100
```

Use the variant option to choose a Whisper variant: (whisper, transformers, whisperx or whisper_timestamped).

```shell
python evaluate_dataset.py cv --variant=whisperx
```

### Evaluating larger files and transcripts

Clean up (normalize) VTT files with vtt_cleaning.py.

```shell
python vtt_cleaning.py
```

Calculate WERs (word error rates) based on reference and hypothesis text files.

```shell
python wer_calculation.py
```
