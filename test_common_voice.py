from datasets import load_dataset

cv_16 = load_dataset('mozilla-foundation/common_voice_16_1', 'de',
                     split='train', trust_remote_code=True)
first = next(iter(cv_16))
print(first)
