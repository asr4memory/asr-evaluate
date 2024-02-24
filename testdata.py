from datasets import load_dataset, Audio, Dataset
from pyarrow import Table

class TestData:
    AUDIO_KEY = "audio"
    TRANSCRIPTION_KEY = "transcription"
    LANGUAGE = "german"

    def __init__(self, length = None):
        self.dataset = Dataset(Table())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __str__(self):
        info = self.dataset.info
        return "{0} {1}".format(info.dataset_name, info.version)


class CommonVoiceTestData(TestData):
    TRANSCRIPTION_KEY = "sentence"

    def __init__(self, length = None):
        self.dataset = load_dataset('mozilla-foundation/common_voice_16_1',
                                    'de',
                                    split='test',
                                    trust_remote_code=True)
        if length: self.dataset = self.dataset.select(range(length))
        # Does this even work?
        self.dataset.cast_column("audio", Audio(sampling_rate=16000))


class FleursTestData(TestData):
    def __init__(self, length = None):
        self.dataset = load_dataset('google/fleurs',
                                    'de_de',
                                    split='test',
                                    trust_remote_code=True)
        if length: self.dataset = self.dataset.select(range(length))
