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

Install mac-only dependencies if you have a Mac:

```shell
pip install -r requirements_mac.txt
```

Create the configuration file.

```shell
cp config.example.toml config.toml
```

## Usage

Right now, only Whisper can be used for evaluation.

### Evaluating datasets

Login with huggingface_hub token, requires a token generated from https://huggingface.co/settings/tokens (not required for custom dataset use).

```shell
huggingface-cli login
```

Use evaluate_dataset.py to evaluate either the Fleurs, the CommonVoice or a locally saved custom dataset.

```shell
python evaluate_dataset.py fleurs
```

Use the length option to reduce the number of data points.

```shell
python evaluate_dataset.py cv --length=100
```

Use the variant option to choose a Whisper variant: whisper (default), transformers, whisperx, whisper_timestamped or whisper_mlx (only compatible with Apple Silicon chips).

```shell
python evaluate_dataset.py cv --variant=whisperx
```

Use the test_size option to split the dataset (default: test_size=0.2). Works only on custom dataset.

```shell
python evaluate_dataset.py custom --test_size=0.4
```

Use the seed option to randomize the test split (default: seed=42). Works only on custom dataset.

```shell
python evaluate_dataset.py custom --seed=84
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
