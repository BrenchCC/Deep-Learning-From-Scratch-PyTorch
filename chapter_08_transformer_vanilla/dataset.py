import random
from typing import Dict, List, Tuple

from torch.utils.data import Dataset

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


class SortDataset(Dataset):
    """
    Toy sequence sorting dataset.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        vocab_size: int = 64,
        min_seq_len: int = 4,
        max_seq_len: int = 12,
        seed: int = 42
    ):
        """
        Initialize sorting dataset.

        Args:
            num_samples (int): Number of synthetic samples.
            vocab_size (int): Vocabulary size including special tokens.
            min_seq_len (int): Minimum sequence length.
            max_seq_len (int): Maximum sequence length.
            seed (int): Random seed.
        """
        if vocab_size <= EOS_TOKEN_ID + 1:
            raise ValueError("vocab_size must be greater than 3.")
        if min_seq_len < 2:
            raise ValueError("min_seq_len must be >= 2.")
        if max_seq_len < min_seq_len:
            raise ValueError("max_seq_len must be >= min_seq_len.")

        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.seed = seed

        self._rng = random.Random(seed)
        self.samples = self._build_samples()

    def _build_samples(self) -> List[List[int]]:
        """
        Build random token sequences.

        Args:
            None

        Returns:
            List[List[int]]: Sequence list.
        """
        samples = []
        for _ in range(self.num_samples):
            seq_len = self._rng.randint(self.min_seq_len, self.max_seq_len)
            tokens = [
                self._rng.randint(EOS_TOKEN_ID + 1, self.vocab_size - 1)
                for _ in range(seq_len)
            ]
            samples.append(tokens)
        return samples

    def __len__(self) -> int:
        """
        Return dataset size.

        Args:
            None

        Returns:
            int: Sample count.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        """
        Get one sample.

        Args:
            index (int): Sample index.

        Returns:
            Dict[str, List[int]]: Seq2seq sample dictionary.
        """
        src = self.samples[index]
        tgt = sorted(src)

        return {
            "src_tokens": src + [EOS_TOKEN_ID],
            "tgt_input_tokens": [BOS_TOKEN_ID] + tgt,
            "tgt_output_tokens": tgt + [EOS_TOKEN_ID]
        }


class ToyTranslationDataset(Dataset):
    """
    Tiny translation dataset loaded from local TSV pairs.
    """

    def __init__(self, data_path: str, repeat_factor: int = 40):
        """
        Initialize translation dataset.

        Args:
            data_path (str): TSV path with `src \t tgt` per line.
            repeat_factor (int): Repeat count to enlarge tiny dataset.
        """
        if repeat_factor < 1:
            raise ValueError("repeat_factor must be >= 1.")

        self.data_path = data_path
        self.repeat_factor = repeat_factor

        raw_pairs = self._load_pairs(data_path)
        self.token_to_id, self.id_to_token = self._build_vocab(raw_pairs)
        self.samples = self._build_samples(raw_pairs)

    def _load_pairs(self, data_path: str) -> List[Tuple[List[str], List[str]]]:
        """
        Load raw translation pairs.

        Args:
            data_path (str): Input file path.

        Returns:
            List[Tuple[List[str], List[str]]]: Tokenized source-target pairs.
        """
        pairs = []
        with open(data_path, "r", encoding = "utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                src_text, tgt_text = line.split("\t")
                src_tokens = src_text.strip().split(" ")
                tgt_tokens = tgt_text.strip().split(" ")
                pairs.append((src_tokens, tgt_tokens))

        if len(pairs) == 0:
            raise ValueError("translation pair file is empty.")

        return pairs

    def _build_vocab(self, pairs: List[Tuple[List[str], List[str]]]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Build joint vocab.

        Args:
            pairs (List[Tuple[List[str], List[str]]]): Raw tokenized pairs.

        Returns:
            Tuple[Dict[str, int], Dict[int, str]]: token_to_id and id_to_token.
        """
        all_tokens = set()
        for src_tokens, tgt_tokens in pairs:
            all_tokens.update(src_tokens)
            all_tokens.update(tgt_tokens)

        sorted_tokens = sorted(list(all_tokens))
        token_to_id = {
            "<PAD>": PAD_TOKEN_ID,
            "<BOS>": BOS_TOKEN_ID,
            "<EOS>": EOS_TOKEN_ID
        }
        for token in sorted_tokens:
            token_to_id[token] = len(token_to_id)

        id_to_token = {idx: token for token, idx in token_to_id.items()}
        return token_to_id, id_to_token

    def _build_samples(self, pairs: List[Tuple[List[str], List[str]]]) -> List[Dict[str, List[int]]]:
        """
        Build encoded samples.

        Args:
            pairs (List[Tuple[List[str], List[str]]]): Raw tokenized pairs.

        Returns:
            List[Dict[str, List[int]]]: Encoded sample list.
        """
        encoded_samples = []
        for _ in range(self.repeat_factor):
            for src_tokens, tgt_tokens in pairs:
                src_ids = [self.token_to_id[token] for token in src_tokens]
                tgt_ids = [self.token_to_id[token] for token in tgt_tokens]

                encoded_samples.append(
                    {
                        "src_tokens": src_ids + [EOS_TOKEN_ID],
                        "tgt_input_tokens": [BOS_TOKEN_ID] + tgt_ids,
                        "tgt_output_tokens": tgt_ids + [EOS_TOKEN_ID]
                    }
                )
        return encoded_samples

    @property
    def vocab_size(self) -> int:
        """
        Return vocabulary size.

        Args:
            None

        Returns:
            int: Vocabulary size.
        """
        return len(self.token_to_id)

    def decode_ids(self, ids: List[int], stop_at_eos: bool = True) -> List[str]:
        """
        Decode id list to token list.

        Args:
            ids (List[int]): Token id list.
            stop_at_eos (bool): Whether to stop at EOS token.

        Returns:
            List[str]: Decoded token list.
        """
        tokens = []
        for idx in ids:
            if idx == PAD_TOKEN_ID:
                continue
            if idx == BOS_TOKEN_ID:
                continue
            if idx == EOS_TOKEN_ID and stop_at_eos:
                break
            if idx == EOS_TOKEN_ID:
                continue
            tokens.append(self.id_to_token.get(idx, "<UNK>"))
        return tokens

    def __len__(self) -> int:
        """
        Return dataset size.

        Args:
            None

        Returns:
            int: Sample count.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        """
        Get one sample.

        Args:
            index (int): Sample index.

        Returns:
            Dict[str, List[int]]: Encoded seq2seq sample.
        """
        return self.samples[index]


class Seq2SeqCollator:
    """
    Collator for seq2seq tasks.
    """

    def __init__(self, pad_token_id: int = PAD_TOKEN_ID):
        """
        Initialize collator.

        Args:
            pad_token_id (int): Padding token id.
        """
        self.pad_token_id = pad_token_id

    def _pad(self, sequences: List[List[int]]) -> List[List[int]]:
        """
        Pad variable-length sequences.

        Args:
            sequences (List[List[int]]): Sequence list.

        Returns:
            List[List[int]]: Padded sequence list.
        """
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            padded.append(seq + [self.pad_token_id] * (max_len - len(seq)))
        return padded

    def __call__(self, batch: List[Dict[str, List[int]]]):
        """
        Collate batch samples.

        Args:
            batch (List[Dict[str, List[int]]]): Raw sample batch.

        Returns:
            Tuple[List[List[int]], List[List[int]], List[List[int]]]: Padded tensors.
        """
        import torch

        src_tokens = [item["src_tokens"] for item in batch]
        tgt_input_tokens = [item["tgt_input_tokens"] for item in batch]
        tgt_output_tokens = [item["tgt_output_tokens"] for item in batch]

        src = torch.tensor(self._pad(src_tokens), dtype = torch.long)
        tgt_in = torch.tensor(self._pad(tgt_input_tokens), dtype = torch.long)
        tgt_out = torch.tensor(self._pad(tgt_output_tokens), dtype = torch.long)
        return src, tgt_in, tgt_out
