import argparse
import json
import re

from jiwer import process_words
from test_datasets import CommonVoiceTestDataset, FleursTestDataset, CustomTestDataset
from whisper_variants import (
    WhisperVariant,
    WhisperTransformersVariant,
    WhisperXVariant,
    WhisperTimestampedVariant,
    WhisperMlxVariant,
    CrisperWhisperVariant
)


def process(dataset, variant, file):
    print(f"Variant: {variant}")
    print(f"Dataset: {dataset}")
    print(f"Evaluating {len(dataset)} data points...")

    actual_list = []
    target_list = []
    output_list = []

    for index in range(len(dataset)):
        try:
            # Attempt to evaluate the data point
            actual, target, metrics = evaluate(dataset, index, variant)

            print("{0} / {1} {2}".format(index + 1, len(dataset), "-" * 70))
            print(actual)
            print(target)
            actual_list.append(actual)
            target_list.append(target)
            print("WER: {:2.1%}".format(metrics.wer))
            output_list.append(
                {
                    "actual": actual,
                    "target": target,
                    "wer": metrics.wer,
                }
            )

        except Exception as e:
            # Catch and print error details, including the index of the problematic file
            print(f"[Error] Failed to process data point at index {index}: {e}")
            # Optionally, log additional data point information
            data_point = dataset[index]
            print(f"  Data point info: {data_point}")
            continue  # Skip to the next data point

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
    result = re.sub(r"[!\?\.,;:]", "", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="asr-evaluate", description="Evaluate automatic speech recognition tools."
    )

    parser.add_argument(
        "dataset",
        choices=["cv", "fleurs", "custom"],
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
        "--seed",
        "-s",
        type=int,
        default=42,
        help="the seed for the random number generator",
    )
    parser.add_argument(
        "--test_size",
        "-t",
        type=float,
        default=0.2,
        help="the fraction of the dataset that should be used for testing",
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
            "crisper_whisper"
        ],
        default="whisper",
        help="the Whisper variant to be evaluated",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("w", encoding="utf-8"),
        help="filename for outputting results as JSON",
    )

    args = parser.parse_args()

    if args.dataset == "cv":
        dataset_class = CommonVoiceTestDataset(args.length)
    elif args.dataset == "fleurs":
        dataset_class = FleursTestDataset(args.length)
    elif args.dataset == "custom":
        dataset_class = CustomTestDataset(args.length, args.test_size, args.seed)

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
    elif args.variant == "crisper_whisper":
        variant = CrisperWhisperVariant()

    output_file = args.output

    process(dataset_class, variant, output_file)
