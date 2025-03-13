import importlib
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import whisper
import whisperx
import whisper_timestamped
from CrisperWhisper.utils import adjust_pauses_for_hf_pipeline_output
from speechbrain.inference import EncoderDecoderASR

mlx_present = bool(importlib.util.find_spec(name="mlx"))
if mlx_present:
    import whisper_mlx


class Variant:
    def transcribe(self, audio, language):
        pass

    def __str__(self):
        return self.__class__.__name__


class WhisperTransformersVariant(Variant):
    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
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

class CrisperWhisperVariant(Variant):
    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "nyrahealth/CrisperWhisper"

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
            chunk_length_s=30,
            batch_size=16,
            return_timestamps='word',
            torch_dtype=torch_dtype,
            device=device,
        )
        self.pipe = pipe

    def transcribe(self, audio, language):
        hf_pipeline_output = self.pipe(audio, generate_kwargs={"language": language})
        result = adjust_pauses_for_hf_pipeline_output(hf_pipeline_output)
        return result["text"]

class WhisperXVariant(Variant):
    def __init__(self):
        self.model = whisperx.load_model("large-v3", "cuda", compute_type="float32")

    def transcribe(self, audio, language):
        audio32 = audio.astype("float32")
        result = self.model.transcribe(audio32, batch_size=7, language=language)
        segments = result["segments"]
        text_parts = [segment["text"] for segment in segments]
        full_text = "".join(text_parts)
        return full_text


class WhisperTimestampedVariant(Variant):
    def __init__(self):
        self.model = whisper_timestamped.load_model("large-v3", "cuda")

    def transcribe(self, audio, language):
        audio32 = audio.astype("float32")
        result = whisper_timestamped.transcribe(
            self.model,
            audio32,
            language=language,
            beam_size=5,
            best_of=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        )

        return result["text"]


class WhisperMlxVariant(Variant):
    """
    WhisperMlxVariant only works when 'mlx' module is installed.
    This module can only be installed on Apple computers.
    """

    def __init__(self):
        if not mlx_present:
            raise RuntimeError(
                "WhisperMlxVariant only works when 'mlx' module is installed."
            )

    def transcribe(self, audio, language):
        audio32 = audio.astype("float32")
        result = whisper_mlx.transcribe(
            audio32,
            path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
            language=language,
        )
        return result["text"]


class WhisperVariant(Variant):
    def __init__(self):
        self.model = whisper.load_model("large-v3", "cuda")

    def transcribe(self, audio, language):
        audio32 = audio.astype("float32")
        result = self.model.transcribe(audio32, language=language)
        return result["text"]

class SpeechbrainVariant(Variant):
    def __init__(self):
        self.model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-conformer-transformerlm-librispeech", savedir="/home/kompiel/.cache/speechbrain/asr-transformer-transformerlm-librispeech")

    def transcribe(self, audio, language):
        result = self.model.transcribe_file(audio)
        return result