from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import argparse

def transcribe_audio(audio_file: str) -> str:
    """
    Lädt das Wav2Vec2-Modell direkt aus Hugging Face und transkribiert die angegebene Audiodatei.
    :param audio_file: Pfad zur WAV-Audiodatei.
    :return: Transkription als String.
    """
    # Modell und Prozessor laden
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-german")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-german")
    
    # Audio laden und vorverarbeiten
    speech_array, sampling_rate = librosa.load(audio_file, sr=16000)
    input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values
    
    # Transkription durchführen
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Dekodieren
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transkribiere eine WAV-Audiodatei mit dem Wav2Vec2-Modell."
    )
    parser.add_argument(
        "audio_file",
        nargs='?',
        default="/Users/pkompiel/python_scripts/asr4memory/asr-evaluate/data/pilot0026_01_01_original_240sec.wav",
        type=str,
        help="Pfad zur WAV-Audiodatei, die transkribiert werden soll."
    )
    args = parser.parse_args()
    result = transcribe_audio(args.audio_file)
    print("Transkription:")
    print(result)