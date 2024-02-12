import random
from datasets import load_dataset

fleurs = load_dataset('google/fleurs', 'de_de', split='train',
                      trust_remote_code=True)
length = len(fleurs)
example = random.choice(fleurs)

print(f"{length} datapoints")
print(f"Example: {example}")
