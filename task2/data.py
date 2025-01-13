import torch
from torch.utils.data import Dataset
import re


def preprocess_text(text):
    text = text.lower().replace("\n", " ")
    text = re.sub(" +", " ", text)
    return ''.join(c for c in text if c.isalpha() or c.isspace())


class TextDataset(Dataset):
    def __init__(self, text, sequence_length, vocab):
        self.text = text
        self.sequence_length = sequence_length
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.encoded_text = [self.char_to_idx[c] for c in text]
        self.inputs, self.targets = self._create_sequences()

    def _create_sequences(self):
        inputs, targets = [], []
        for i in range(len(self.encoded_text) - self.sequence_length):
            inputs.append(self.encoded_text[i:i + self.sequence_length])
            targets.append(self.encoded_text[i + self.sequence_length])
        return inputs, targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])
