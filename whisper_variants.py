import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import whisperx

class Variant:
    def transcribe(self, audio, language):
        pass

    def __str__(self):
        return self.__class__.__name__


class WhisperTransformersVariant(Variant):
    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = (torch.float16
                       if torch.cuda.is_available() else torch.float32)

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
            use_safetensors=True
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
        self.pipe = pipe

    def transcribe(self, audio, language):
        result = self.pipe(audio, generate_kwargs={"language": language})
        return result["text"]


class WhisperXVariant(Variant):
    def __init__(self):
        self.model = whisperx.load_model("large-v3",
                                         "cpu",
                                         compute_type="float32")

    def transcribe(self, audio, language):
        audio32 = audio.astype("float32")
        result = self.model.transcribe(audio32,
                                       batch_size=2,
                                       language=language)
        segments = result["segments"]
        text_parts = [segment["text"] for segment in segments]
        full_text = "".join(text_parts)
        return full_text
