import logging
import random

import torch
from torch.utils.data import Dataset

logger = logging.getLogger("SyntheticNameDataset")


class SyntheticNameDataset(Dataset):
    """
    Synthetic dataset that classifies names into language-like categories.
    """

    def __init__(self, num_samples = 100000):
        """
        Initialize dataset and generate synthetic samples.

        Args:
            num_samples (int): Number of synthetic samples.
        """
        self.data = []
        self.labels = []
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.classes = ["Russian", "English", "Italian", "Japanese", "Greek"]
        self._generate_data(num_samples)

    def _generate_data(self, num_samples):
        """
        Generate synthetic names by class-specific suffix rules.

        Args:
            num_samples (int): Number of samples to generate.
        """
        rules = {
            0: ["ov", "sky", "ev", "in", "ka"],
            1: ["son", "man", "er", "ton", "ley"],
            2: ["ini", "elli", "ano", "rio", "ucci"],
            3: ["yama", "to", "wa", "ki", "ko"],
            4: ["os", "is", "as", "opoulos", "akis"]
        }
        chars = "abcdefghijklmnopqrstuvwxyz"

        logger.info(f"Generating {num_samples} synthetic samples...")
        for _ in range(num_samples):
            label = random.choice(list(rules.keys()))
            base_len = random.randint(3, 8)
            base_name = "".join([random.choice(chars) for _ in range(base_len)])
            suffix = random.choice(rules[label])

            full_name = base_name + suffix
            self.data.append(full_name)
            self.labels.append(label)

            for char in full_name:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)

        logger.info(f"Finished dataset generation with {len(self.vocab)} unique characters")

    def __len__(self):
        """
        Return dataset size.

        Args:
            None

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get one sample.

        Args:
            idx (int): Sample index.

        Returns:
            tuple: (raw_name, label_index)
        """
        return self.data[idx], self.labels[idx]

    def save_vocab(self, path = "chapter_06_rnn_lstm_seq/data/vocab.txt"):
        """
        Save vocabulary to disk.

        Args:
            path (str): Output file path.

        Returns:
            None
        """
        with open(path, "w", encoding = "utf-8") as f:
            for char, idx in self.vocab.items():
                f.write(f"{char}\t{idx}\n")

    def save_data(self, path = "chapter_06_rnn_lstm_seq/data/synthetic_names.txt"):
        """
        Save generated synthetic samples to disk.

        Args:
            path (str): Output file path.

        Returns:
            None
        """
        with open(path, "w", encoding = "utf-8") as f:
            for name, label in zip(self.data, self.labels):
                f.write(f"{name}\t{self.classes[label]}\n")


class VectorizedCollator:
    """
    Collator that vectorizes raw name strings and pads variable-length sequences.
    """

    def __init__(self, vocab):
        """
        Initialize collator.

        Args:
            vocab (dict): Vocabulary mapping.
        """
        self.vocab = vocab

    def __call__(self, batch):
        """
        Convert raw text batch into padded tensor batch.

        Args:
            batch (list): List of ``(text, label)`` tuples.

        Returns:
            tuple: (padded_sequences, lengths, labels)
        """
        texts, labels = zip(*batch)

        sequences = []
        lengths = []
        for text in texts:
            seq = [self.vocab.get(token, self.vocab["<unk>"]) for token in text]
            sequences.append(torch.tensor(seq, dtype = torch.long))
            lengths.append(len(seq))

        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first = True,
            padding_value = self.vocab["<pad>"]
        )

        lengths = torch.tensor(lengths, dtype = torch.long)
        labels = torch.tensor(labels, dtype = torch.long)
        return padded_sequences, lengths, labels


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    dataset = SyntheticNameDataset()
    dataset.save_vocab()
    dataset.save_data()
