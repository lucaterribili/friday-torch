from torch.utils.data import Dataset
import torch

class FridayDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, tokenizer, max_len):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.filter_invalid_pairs()

    def __len__(self):
        return len(self.source_sentences)

    def old_pad_sequence(self, original_source, original_target):
        source = self.tokenizer.encode(f"{original_source[0]}: {original_source[1]}")
        target = self.tokenizer.encode(original_target)
        target.insert(0, self.tokenizer.bos_id())
        target.append(self.tokenizer.eos_id())

        # Applica padding
        target_len = len(target)
        source_len = len(source)

        remaining_padding_for_target = self.max_len - target_len
        if remaining_padding_for_target > 0:
            target = target[:-1] + [self.tokenizer.pad_id()] * remaining_padding_for_target + [self.tokenizer.eos_id()]

        remaining_padding_for_source = self.max_len - source_len
        if remaining_padding_for_source > 0:
            source = source + [self.tokenizer.pad_id()] * remaining_padding_for_source

        return source, target

    def pad_sequence(self, original_source, original_target):
        # Encode the source and target sequences
        source = self.tokenizer.encode(f"{original_source[0]}: {original_source[1]}")
        target = self.tokenizer.encode(original_target)

        # Insert BOS token at the beginning of the target sequence
        target.insert(0, self.tokenizer.bos_id())
        target.append(self.tokenizer.eos_id())  # Add EOS token at the end

        # Calculate lengths
        source_len = len(source)
        target_len = len(target)

        # Padding for target sequence (add padding after EOS)
        remaining_padding_for_target = self.max_len - target_len
        if remaining_padding_for_target > 0:
            target = target + [self.tokenizer.pad_id()] * remaining_padding_for_target

        # Padding for source sequence
        remaining_padding_for_source = self.max_len - source_len
        if remaining_padding_for_source > 0:
            source = source + [self.tokenizer.pad_id()] * remaining_padding_for_source

        return source, target

    def filter_invalid_pairs(self):
        valid_sources = []
        valid_targets = []
        for src, trg in zip(self.source_sentences, self.target_sentences):
            padded_src, padded_trg = self.pad_sequence(src, trg)
            # Verifica che entrambi siano validi
            if len(padded_src) <= self.max_len and len(padded_trg) <= self.max_len:
                valid_sources.append(src)
                valid_targets.append(trg)

        # Aggiorna le liste con i dati validi
        self.source_sentences = valid_sources
        self.target_sentences = valid_targets

    def __getitem__(self, idx):
        src = self.source_sentences[idx]
        trg = self.target_sentences[idx]
        # Converti le frasi in indici e aggiungi il padding
        src_indexes, trg_indexes = self.pad_sequence(src, trg)
        return torch.tensor(src_indexes), torch.tensor(trg_indexes)