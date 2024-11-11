import argparse
import re
import os

from jiwer import process_words
from test_datasets import CommonVoiceTestDataset, FleursTestDataset
from whisper_variants import (
    WhisperVariant,
    WhisperTransformersVariant,
    WhisperXVariant,
    WhisperTimestampedVariant,
    WhisperMlxVariant,
)
import json

def process(dataset, variant, args):
    print(f"Variant: {variant}")
    print(f"Dataset: {dataset}")
    print(f"Evaluating {len(dataset)} data points...")

    actual_list = []
    target_list = []
    # List to store results
    results = []

    for index in range(len(dataset)):
        actual, target, metrics = evaluate(dataset, index, variant)

        print("{0} / {1} {2}".format(index + 1, len(dataset), "-" * 70))
        print(actual)
        print(target)
        actual_list.append(actual)
        target_list.append(target)

        # Save the result as a dictionary in the list
        result = {
            "actual": actual,
            "target": target,
            "WER": metrics.wer
        }
        results.append(result)

        print("WER: {:2.1%}".format(metrics.wer))

        # Save the list as a JSON file after every 100 steps
        if index % 100 == 0:
            results_dir = r"/home/chrvt/asr-evaluate" #os.path.abspath("results")
            with open(os.path.join(results_dir, args.variant + "_" + args.dataset + "_" + str(index) + ".json"), "w") as outfile:
                json.dump(results, outfile, indent=4)

    combined_metrics = process_words(
        " ".join(target_list),
        " ".join(actual_list),
    )
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

    args = parser.parse_args()

    if args.dataset == "cv":
        dataset_class = CommonVoiceTestDataset
    elif args.dataset == "fleurs":
        dataset_class = FleursTestDataset

    length = args.length
    dataset = dataset_class(length)
    
    print('choose variant')

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

    process(dataset, variant, args)
