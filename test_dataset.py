import os
import numpy as np
import sentencepiece as spm
import random

# Setup path
base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, "multitask_dataset")
tokenizer_model = os.path.join(base_dir, "models", "tokenizer", "model.model")

# Load datasets
train_data = np.load(os.path.join(dataset_path, "dataset.npy"), allow_pickle=True)
val_data = np.load(os.path.join(dataset_path, "dataset_validation.npy"), allow_pickle=True)

# Carica tokenizer (presupponiamo che sia un modello .model nello stesso folder)
sp = spm.SentencePieceProcessor()
sp.Load(tokenizer_model)

def sample_and_decode(data, name):
    print(f"\n--- {name} ---")
    indices = list(range(len(data) - 10, len(data)))
    for i in indices:
        src, trg = data[i]
        decoded_src = sp.Decode(src)
        decoded_trg = sp.Decode(trg)
        print(f"\n[{i}] {name}")
        print(f"Input:  {decoded_src}")
        print(f"Output: {decoded_trg}")


# Esegui il controllo
sample_and_decode(train_data, "TRAIN")
sample_and_decode(val_data, "VALIDATION")
