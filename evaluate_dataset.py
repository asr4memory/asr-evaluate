import argparse
import re
import statistics
from jiwer import process_words
from test_datasets import CommonVoiceTestDataset, FleursTestDataset
from whisper_variants import (
    WhisperTransformersVariant,
    WhisperXVariant,
    WhisperTimestampedVariant)

def normalize(text):
    result = text.strip().lower()
    result = re.sub(r'[!\?\.,;]', '', result)
    return result


def evaluate(dataset, index, variant):
    data_point = dataset[index]
    sample = data_point[dataset.AUDIO_KEY]["array"]
    actual = normalize(variant.transcribe(audio=sample,
                                          language=dataset.LANGUAGE))
    target = normalize(data_point[dataset.TRANSCRIPTION_KEY])
    metrics = process_words(target, actual)
    return (actual, target, metrics)


def process(dataset, variant):
    print(f"Variant: {variant}")
    print(f"Dataset: {dataset}")
    print(f"Evaluating {len(dataset)} data points...")

    wer_list = []

    for index in range(len(dataset)):
        actual, target, metrics = evaluate(dataset, index, variant)

        print("{0} / {1} {2}".format(index + 1,
                                    len(dataset),
                                    '-' * 70))
        print(actual)
        print(target)
        wer_list.append(metrics.wer)
        print("WER: {:2.1%}".format(metrics.wer))

    mean = statistics.mean(wer_list)
    print("Average WER of {0:2.1%} for {1} data points".format(
        mean, len(dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="asr-evaluate",
        description="Evaluate automatic speech recognition tools.")

    parser.add_argument('dataset',
                        choices=['cv', 'fleurs'],
                        help='the name of the dataset the tool is evaluated on')
    parser.add_argument('--length',
                        type=int,
                        default=None,
                        help='the number of data points that should be used')
    parser.add_argument('--variant',
                        choices=['transformers', 'whisperx', 'whisper_timestamped'],
                        default='transformers',
                        help='the Whisper variant to be evaluated')

    args = parser.parse_args()

    if args.dataset == 'cv':
        dataset_class = CommonVoiceTestDataset
    elif args.dataset == 'fleurs':
        dataset_class = FleursTestDataset

    length = args.length
    dataset = dataset_class(length)

    if args.variant == 'whisperx':
        variant = WhisperXVariant()
    elif args.variant == 'transformers':
        variant = WhisperTransformersVariant()
    elif args.variant == 'whisper_timestamped':
        variant = WhisperTimestampedVariant()

    process(dataset, variant)
