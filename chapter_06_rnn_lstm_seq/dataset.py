import random
import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("Synthetic Dataset")
# --- Dataset Section (Synthetic) ---
# 为了方便演示，我们创建一个合成数据集：分类名字属于哪种语言
# 规则（简化版）：
# 以 'ov', 'sky' 结尾 -> Russian (0)
# 以 'mc', 'son' 结尾 -> English (1)
# 以 'i', 'o' 结尾 -> Italian (2)
# 以 'yama', 'to', 'wa', 'ki', 'ko' 结尾 -> Japanese (3)
# 以 'os', 'is', 'as', 'opoulos', 'akis' 结尾 -> Greek (4)
# 其他 -> Unknown (3)
class SyntheticNameDataset(Dataset):
    def __init__(self, num_samples = 100000):
        self.data = []
        self.labels = []
        self.vocab = {'<pad>': 0, '<unk>': 1}
        self.classes = ['Russian', 'English', 'Italian', 'Japanese', 'Greek']
        
        self._generate_data(num_samples)
        
    def _generate_data(self, num_samples):
        # Define generation rules (suffixes/patterns)
        rules = {
            0: ['ov', 'sky', 'ev', 'in', 'ka'],          # Russian
            1: ['son', 'man', 'er', 'ton', 'ley'],       # English
            2: ['ini', 'elli', 'ano', 'rio', 'ucci'],    # Italian
            3: ['yama', 'to', 'wa', 'ki', 'ko'],         # Japanese
            4: ['os', 'is', 'as', 'opoulos', 'akis']     # Greek
        }
        chars = "abcdefghijklmnopqrstuvwxyz"
        
        logger.info(f"Generating {num_samples} samples...")
        for _ in range(num_samples):
            label = random.choice(list(rules.keys()))
            # Random base length 3-8
            base_len = random.randint(3, 8)
            base_name = "".join([random.choice(chars) for _ in range(base_len)])
            suffix = random.choice(rules[label])
            
            full_name = base_name + suffix
            self.data.append(full_name)
            self.labels.append(label)
            
            # Update vocab on the fly
            for char in full_name:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
        logger.info(f"Generated {num_samples} samples with {len(self.vocab)} unique characters")
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Return raw text and label, vectorization happens in collate or manually
        return self.data[idx], self.labels[idx]

    def save_vocab(self, path = 'chapter_06_rnn_lstm_seq/data/vocab.txt'):
        with open(path, 'w') as f:
            for char, idx in self.vocab.items():
                f.write(f"{char}\t{idx}\n")
    
    def save_data(self, path = 'chapter_06_rnn_lstm_seq/data/synthetic_names.txt'):
        with open(path, 'w') as f:
            for name, label in zip(self.data, self.labels):
                f.write(f"{name}\t{self.classes[label]}\n")

class VectorizedCollator:
    def __init__(self, vocab):
        self.vocab = vocab  
    
    def __call__(self, batch):
        """
        1. 将文本转换为索引
        2. 计算长度
        3. Padding
        """
        # 提取文本和标签
        texts, labels = zip(*batch)

        sequences = []
        lengths = []
        # Vectorize texts
        for text in texts:
            seq = [self.vocab.get(token, self.vocab['<unk>']) for token in text]
            sequences.append(torch.tensor(seq))
            lengths.append(len(seq))
        
        # Padding
        padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first = True, padding_value = self.vocab['<pad>'])

        # 转换为Tensor
        lengths = torch.tensor(lengths)
        labels = torch.tensor(labels)
        
        return padded_sequences, lengths, labels
    
if __name__ == '__main__':
    dataset = SyntheticNameDataset()
    dataset.save_vocab()
    dataset.save_data()