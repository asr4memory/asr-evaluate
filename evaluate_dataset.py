import argparse
import json
import re

from jiwer import process_words
from test_datasets import CommonVoiceTestDataset, FleursTestDataset
from whisper_variants import (
    WhisperVariant,
    WhisperTransformersVariant,
    WhisperXVariant,
    WhisperTimestampedVariant,
    WhisperMlxVariant,
)


def process(dataset, variant, file):
    print(f"Variant: {variant}")
    print(f"Dataset: {dataset}")
    print(f"Evaluating {len(dataset)} data points...")

    actual_list = []
    target_list = []
    output_list = []

    for index in range(len(dataset)):
        actual, target, metrics = evaluate(dataset, index, variant)

        print("{0} / {1} {2}".format(index + 1, len(dataset), "-" * 70))
        print(actual)
        print(target)
        actual_list.append(actual)
        target_list.append(target)
        print("WER: {:2.1%}".format(metrics.wer))
        output_list.append({
            "actual": actual,
            "target": target,
            "wer": metrics.wer,
        })

    combined_metrics = process_words(
        " ".join(target_list),
        " ".join(actual_list),
    )

    if file:
        json.dump(output_list, file, indent=4, ensure_ascii=False)
        file.close()

    print(
        "Average WER of {0:2.1%} for {1} data points".format(
            combined_metrics.wer, len(dataset)
        )
    )


def evaluate(dataset, index, variant):
    data_point = dataset[index]
    sample = data_point[dataset.AUDIO_KEY]["array"]
    actual = normalize(variant.transcribe(audio=sample, language=dataset.LANGUAGE))
    target = normalize(data_point[dataset.TRANSCRIPTION_KEY])
    metrics = process_words(target, actual)
    return (actual, target, metrics)


def normalize(text):
    result = text.strip().lower()
    result = re.sub(r"[!\?\.,;]", "", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="asr-evaluate", description="Evaluate automatic speech recognition tools."
    )

    parser.add_argument(
        "dataset",
        choices=["cv", "fleurs"],
        help="the name of the dataset the tool is evaluated on",
    )
    parser.add_argument(
        "--length",
        "--limit",
        "-l",
        type=int,
        default=None,
        help="the number of data points that should be used",
    )
    parser.add_argument(
        "--variant",
        "-v",
        choices=[
            "whisper",
            "transformers",
            "whisperx",
            "whisper_timestamped",
            "whisper_mlx",
        ],
        default="whisper",
        help="the Whisper variant to be evaluated",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("w", encoding="utf-8"),
        help="filename for outputting results as JSON"
    )

    args = parser.parse_args()

    if args.dataset == "cv":
        dataset_class = CommonVoiceTestDataset
    elif args.dataset == "fleurs":
        dataset_class = FleursTestDataset

    length = args.length
    dataset = dataset_class(length)

    if args.variant == "whisper":
        variant = WhisperVariant()
    elif args.variant == "whisperx":
        variant = WhisperXVariant()
    elif args.variant == "transformers":
        variant = WhisperTransformersVariant()
    elif args.variant == "whisper_timestamped":
        variant = WhisperTimestampedVariant()
    elif args.variant == "whisper_mlx":
        variant = WhisperMlxVariant()

    output_file = args.output

    process(dataset, variant, output_file)
