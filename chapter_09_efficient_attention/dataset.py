import logging
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


class ToyNextTokenDataset(Dataset):
    """
    Synthetic next-token prediction dataset.
    """

    def __init__(
        self,
        num_samples: int = 2000,
        vocab_size: int = 128,
        seq_len: int = 64,
        seed: int = 42
    ):
        """
        Initialize toy dataset.

        Args:
            num_samples (int): Number of samples.
            vocab_size (int): Vocabulary size including special tokens.
            seq_len (int): Token length before adding BOS/EOS.
            seed (int): Random seed.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0.")
        if vocab_size <= EOS_TOKEN_ID + 1:
            raise ValueError("vocab_size must be greater than 3.")
        if seq_len < 4:
            raise ValueError("seq_len must be >= 4.")

        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed

        self._rng = random.Random(seed)
        self.samples = self._build_samples()
        logger.info(f"Built ToyNextTokenDataset with {len(self.samples)} samples")

    def _build_samples(self) -> List[List[int]]:
        """
        Build token sequences with BOS/EOS.

        Args:
            None

        Returns:
            List[List[int]]: Sequence list.
        """
        samples = []
        for _ in range(self.num_samples):
            middle_tokens = [
                self._rng.randint(EOS_TOKEN_ID + 1, self.vocab_size - 1)
                for _ in range(self.seq_len)
            ]
            sequence = [BOS_TOKEN_ID] + middle_tokens + [EOS_TOKEN_ID]
            samples.append(sequence)
        return samples

    def __len__(self) -> int:
        """
        Return sample count.

        Args:
            None

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        """
        Return one sequence sample.

        Args:
            index (int): Sample index.

        Returns:
            Dict[str, List[int]]: Sample dictionary.
        """
        return {"token_sequence": self.samples[index]}


class ToyNextTokenCollator:
    """
    Collator for next-token prediction.
    """

    def __init__(self, pad_token_id: int = PAD_TOKEN_ID):
        """
        Initialize collator.

        Args:
            pad_token_id (int): Padding token id.
        """
        self.pad_token_id = pad_token_id

    def _pad_int_sequences(self, sequences: List[List[int]]) -> torch.Tensor:
        """
        Pad integer sequences.

        Args:
            sequences (List[List[int]]): Input sequence list.

        Returns:
            torch.Tensor: Padded tensor [B, S].
        """
        max_len = max(len(sequence) for sequence in sequences)
        padded = []
        for sequence in sequences:
            pad_len = max_len - len(sequence)
            padded.append(sequence + [self.pad_token_id] * pad_len)
        return torch.tensor(padded, dtype = torch.long)

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build shifted next-token training tensors.

        Args:
            batch (List[Dict[str, List[int]]]): Raw batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                input_ids, labels, padding_mask.
        """
        token_sequences = [sample["token_sequence"] for sample in batch]
        padded = self._pad_int_sequences(token_sequences)

        input_ids = padded[:, :-1]
        labels = padded[:, 1:]
        padding_mask = input_ids.eq(self.pad_token_id)

        return input_ids, labels, padding_mask
