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


cv = CommonVoiceTestData(20)
fleurs = FleursTestData(20)
data = cv

wer_list = []


def evaluate(data_point):
    sample = data_point[data.AUDIO_KEY]
    result = pipe(sample, generate_kwargs={"language": data.LANGUAGE})
    actual = normalize(result["text"])
    target = normalize(data_point[data.TRANSCRIPTION_KEY])
    metrics = process_words(target, actual)
    return (actual, target, metrics)


print(f"Dataset: {data}")
print(f"Evaluating {len(data)} data points...")

# TODO: Use this later.
#for out in pipe(KeyDataset(selected_common_voice, "audio"),
#                     generate_kwargs={"language": "german"}):
#    print(out["text"])


for index in range(len(data)):
    actual, target, metrics = evaluate(data[index])
    print("{0} / {1} {2}".format(index + 1,
                                 len(data),
                                 '-' * 70))
    print(actual)
    print(target)
    wer_list.append(metrics.wer)
    print("WER: {:2.1%}".format(metrics.wer))

mean = statistics.mean(wer_list)
print("Average WER of {0:2.1%} for {1} data points".format(
    mean, len(data)))
