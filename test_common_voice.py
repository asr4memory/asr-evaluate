import random
from datasets import load_dataset

cv_16 = load_dataset('mozilla-foundation/common_voice_16_1', 'de',
                     split='train', trust_remote_code=True)
length = len(cv_16)
example = random.choice(cv_16)

print(f"{length} datapoints")
print(f"Example: {example}")
