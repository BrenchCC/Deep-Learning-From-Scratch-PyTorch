import logging
import random
from typing import List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


class ToyCopyDataset(Dataset):
    """
    Synthetic sequence dataset for seq2seq copy task.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        vocab_size: int = 64,
        min_seq_len: int = 4,
        max_seq_len: int = 20,
        seed: int = 42
    ):
        """
        Initialize the toy copy dataset.

        Args:
            num_samples (int): Number of generated samples.
            vocab_size (int): Vocabulary size including special tokens.
            min_seq_len (int): Minimum sequence length.
            max_seq_len (int): Maximum sequence length.
            seed (int): Random seed for reproducible generation.
        """
        if vocab_size <= EOS_TOKEN_ID + 1:
            raise ValueError("vocab_size must be greater than 3.")
        if min_seq_len < 1:
            raise ValueError("min_seq_len must be >= 1.")
        if max_seq_len < min_seq_len:
            raise ValueError("max_seq_len must be >= min_seq_len.")

        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.seed = seed

        self._rng = random.Random(seed)
        self.samples = self._build_samples()
        logger.info(f"Built ToyCopyDataset with {len(self.samples)} samples")

    def _build_samples(self) -> List[List[int]]:
        """
        Build random token sequences for copy task.

        Args:
            None

        Returns:
            List[List[int]]: Generated token sequences without BOS/EOS.
        """
        samples = []
        for _ in range(self.num_samples):
            seq_len = self._rng.randint(self.min_seq_len, self.max_seq_len)
            token_ids = [
                self._rng.randint(EOS_TOKEN_ID + 1, self.vocab_size - 1)
                for _ in range(seq_len)
            ]
            samples.append(token_ids)
        return samples

    def __len__(self) -> int:
        """
        Return the dataset size.

        Args:
            None

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> List[int]:
        """
        Return one token sequence.

        Args:
            index (int): Sample index.

        Returns:
            List[int]: Token sequence without special tokens.
        """
        return self.samples[index]


class ToyCopyCollator:
    """
    Convert raw token sequences into padded src/tgt tensors.
    """

    def __init__(
        self,
        pad_token_id: int = PAD_TOKEN_ID,
        bos_token_id: int = BOS_TOKEN_ID,
        eos_token_id: int = EOS_TOKEN_ID
    ):
        """
        Initialize collator with special token ids.

        Args:
            pad_token_id (int): Padding token id.
            bos_token_id (int): Begin-of-sequence token id.
            eos_token_id (int): End-of-sequence token id.
        """
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def _pad_sequences(self, sequences: List[List[int]]) -> torch.Tensor:
        """
        Pad variable-length sequences to a tensor.

        Args:
            sequences (List[List[int]]): Input token sequences.

        Returns:
            torch.Tensor: Padded tensor with shape [batch, max_len].
        """
        max_len = max(len(item) for item in sequences)
        padded = []
        for item in sequences:
            padded_item = item + [self.pad_token_id] * (max_len - len(item))
            padded.append(padded_item)
        return torch.tensor(padded, dtype = torch.long)

    def __call__(self, batch: List[List[int]]):
        """
        Build src, decoder-input, and decoder-target tensors.

        Args:
            batch (List[List[int]]): Raw token sequences.

        Returns:
            tuple: (src_tokens, tgt_input_tokens, tgt_output_tokens).
        """
        src_sequences = []
        tgt_input_sequences = []
        tgt_output_sequences = []

        for tokens in batch:
            src_sequences.append(tokens + [self.eos_token_id])
            tgt_input_sequences.append([self.bos_token_id] + tokens)
            tgt_output_sequences.append(tokens + [self.eos_token_id])

        src_tokens = self._pad_sequences(src_sequences)
        tgt_input_tokens = self._pad_sequences(tgt_input_sequences)
        tgt_output_tokens = self._pad_sequences(tgt_output_sequences)
        return src_tokens, tgt_input_tokens, tgt_output_tokens


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )

    dataset = ToyCopyDataset(num_samples = 3, vocab_size = 16, min_seq_len = 3, max_seq_len = 5, seed = 123)
    collator = ToyCopyCollator()
    batch = [dataset[0], dataset[1], dataset[2]]
    src, tgt_in, tgt_out = collator(batch)
    logger.info(f"src shape: {src.shape}")
    logger.info(f"tgt_in shape: {tgt_in.shape}")
    logger.info(f"tgt_out shape: {tgt_out.shape}")
