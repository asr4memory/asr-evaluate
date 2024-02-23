import re
import statistics
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Audio
from jiwer import process_words

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


def evaluate(data_point):
    sample = data_point["audio"]
    result = pipe(sample)
    actual = normalize(result["text"])
    target = normalize(data_point["sentence"])
    metrics = process_words(target, actual)
    return (actual, target, metrics)


common_voice = load_dataset('mozilla-foundation/common_voice_16_1',
                            'de',
                            split='test',
                            trust_remote_code=True)

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


data_point_count = 20
wer_list = []

print("Evaluating {0} of {1} data points...".format(data_point_count,
                                                    len(common_voice)))

for index in range(data_point_count):
    actual, target, metrics = evaluate(common_voice[index])
    print("{0} / {1} {2}".format(index + 1, data_point_count, '-' * 70))
    print(actual)
    print(target)
    wer_list.append(metrics.wer)
    print("WER: {:2.1%}".format(metrics.wer))

mean = statistics.mean(wer_list)
print("Average WER of {0:2.1%} for {1} data points".format(mean,
                                                           data_point_count))
