import logging
import random
from typing import Dict
from typing import List
from typing import Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1
UNK_TOKEN_ID = 2


class MaskedCopyDataset(Dataset):
    """
    Synthetic dataset for masked copy reconstruction.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        vocab_size: int = 64,
        min_seq_len: int = 6,
        max_seq_len: int = 20,
        mask_ratio: float = 0.3,
        seed: int = 42
    ):
        """
        Initialize dataset.

        Args:
            num_samples (int): Number of synthetic samples.
            vocab_size (int): Vocabulary size including special tokens.
            min_seq_len (int): Minimum raw sequence length.
            max_seq_len (int): Maximum raw sequence length.
            mask_ratio (float): Ratio of masked positions.
            seed (int): Random seed.
        """
        if vocab_size <= UNK_TOKEN_ID + 1:
            raise ValueError("vocab_size must be greater than 3.")
        if min_seq_len < 2:
            raise ValueError("min_seq_len must be >= 2.")
        if max_seq_len < min_seq_len:
            raise ValueError("max_seq_len must be >= min_seq_len.")
        if mask_ratio <= 0.0 or mask_ratio >= 1.0:
            raise ValueError("mask_ratio must be in (0, 1).")

        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        self.seed = seed

        self._rng = random.Random(seed)
        self.samples = self._build_samples()
        logger.info(f"Built MaskedCopyDataset with {len(self.samples)} samples")

    def _build_samples(self) -> List[List[int]]:
        """
        Build random token samples.

        Args:
            None

        Returns:
            List[List[int]]: Raw token sequence list.
        """
        token_samples = []
        for _ in range(self.num_samples):
            seq_len = self._rng.randint(self.min_seq_len, self.max_seq_len)
            seq = [
                self._rng.randint(UNK_TOKEN_ID + 1, self.vocab_size - 1)
                for _ in range(seq_len)
            ]
            token_samples.append(seq)
        return token_samples

    def __len__(self) -> int:
        """
        Return dataset size.

        Args:
            None

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        """
        Return one masked copy sample.

        Args:
            index (int): Sample index.

        Returns:
            Dict[str, List[int]]: Input/target/predict mask lists.
        """
        target_tokens = self.samples[index]
        input_tokens = list(target_tokens)

        num_to_mask = max(1, int(round(len(target_tokens) * self.mask_ratio)))
        chosen_positions = self._rng.sample(list(range(len(target_tokens))), num_to_mask)

        predict_mask = [False] * len(target_tokens)
        for pos in chosen_positions:
            input_tokens[pos] = MASK_TOKEN_ID
            predict_mask[pos] = True

        return {
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "predict_mask": predict_mask
        }


class MaskedCopyCollator:
    """
    Collator for masked copy dataset.
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
            sequences (List[List[int]]): Sequence list.

        Returns:
            torch.Tensor: Padded tensor [batch, max_len].
        """
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            padded.append(seq + [self.pad_token_id] * (max_len - len(seq)))
        return torch.tensor(padded, dtype = torch.long)

    def _pad_bool_sequences(self, sequences: List[List[bool]]) -> torch.Tensor:
        """
        Pad boolean sequences.

        Args:
            sequences (List[List[bool]]): Boolean sequence list.

        Returns:
            torch.Tensor: Padded bool tensor [batch, max_len].
        """
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            padded.append(seq + [False] * (max_len - len(seq)))
        return torch.tensor(padded, dtype = torch.bool)

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate batch into tensors.

        Args:
            batch (List[Dict[str, List[int]]]): Raw batch samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                input_tokens, target_tokens, predict_mask.
        """
        input_tokens = [item["input_tokens"] for item in batch]
        target_tokens = [item["target_tokens"] for item in batch]
        predict_masks = [item["predict_mask"] for item in batch]

        input_tensor = self._pad_int_sequences(input_tokens)
        target_tensor = self._pad_int_sequences(target_tokens)
        predict_tensor = self._pad_bool_sequences(predict_masks)
        return input_tensor, target_tensor, predict_tensor
