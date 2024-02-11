from datasets import load_dataset

fleurs = load_dataset('google/fleurs', 'de_de', split='train',
                      trust_remote_code=True)
first = next(iter(fleurs))
print(first)
