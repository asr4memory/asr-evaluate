import argparse
import re
import statistics
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
#from tqdm.auto import tqdm
#from transformers.pipelines.pt_utils import KeyDataset
from jiwer import process_words
from testdata import CommonVoiceTestData, FleursTestData


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=2,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)


def normalize(text):
    result = text.strip().lower()
    result = re.sub(r'[!\?\.,;]', '', result)
    return result


def evaluate(data_point, dataset):
    sample = data_point[dataset.AUDIO_KEY]
    result = pipe(sample, generate_kwargs={"language": dataset.LANGUAGE})
    actual = normalize(result["text"])
    target = normalize(data_point[dataset.TRANSCRIPTION_KEY])
    metrics = process_words(target, actual)
    return (actual, target, metrics)


def process(dataset):
    print(f"Dataset: {dataset}")
    print(f"Evaluating {len(dataset)} data points...")

    # TODO: Use this later.
    #for out in pipe(KeyDataset(selected_common_voice, "audio"),
    #                     generate_kwargs={"language": "german"}):
    #    print(out["text"])

    wer_list = []

    for index in range(len(dataset)):
        actual, target, metrics = evaluate(dataset[index], dataset)
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

    args = parser.parse_args()

    if args.dataset == 'cv':
        dataset_class = CommonVoiceTestData
    elif args.dataset == 'fleurs':
        dataset_class = FleursTestData

    length = args.length

    dataset = dataset_class(length)
    process(dataset)
