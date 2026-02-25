import os
import sys
import random 
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.append(os.getcwd())

from chapter_06_rnn_lstm_seq.model import DynamicRNNClassifier
from chapter_06_rnn_lstm_seq.dataset import SyntheticNameDataset, VectorizedCollator

from utils import get_device, setup_seed, Timer, save_json, count_parameters, log_model_info

logger = logging.getLogger("RNN_Experiment")

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch_idx,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc = f"Epoch {epoch_idx}", leave = False)
    
    for inputs, lengths, targets in loop:
        inputs = inputs.to(device)
        lengths = lengths.to(torch.device("cpu")) # lengths stays on CPU for pack_padded_sequence
        targets = targets.to(device)
        
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)

        # Backward
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        
        optimizer.step()
        
        # Stats
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Update progress bar
        loop.set_postfix(loss = loss.item(), acc = correct / total)

        return running_loss / len(dataloader), correct / total

def evaluate_inference(
    model,
    vocab,
    classes,
    device,
    samples
):
    model.eval()
    results = []

    logger.info("Evaluating inference...")
    with torch.no_grad():
        for name in samples:
            indice = [vocab.get(token, vocab['<unk>']) for token in name]
            tensor_in = torch.tensor(indice).unsqueeze(0).to(device)
            length = torch.tensor([len(indice)])    
            
            output = model(tensor_in, length)
            pred_idx = torch.argmax(output, dim = 1).item()
            pred_class = classes[pred_idx]
            
            print(f"Input: {name:15s} | Prediction: {pred_class}")
            results.append((name, pred_class))

def args_parser():
    parser = argparse.ArgumentParser(description = "Train RNN/LSTM/GRU on Synthetic Data")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
    parser.add_argument("--epochs", type = int, default = 10, help = "Number of training epochs")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size")
    parser.add_argument("--hidden_dim", type = int, default = 256, help = "Hidden dimension size")
    parser.add_argument("--embed_dim", type = int, default = 128, help = "Embedding dimension size")
    parser.add_argument("--lr", type = float, default = 0.001, help = "Learning rate")
    parser.add_argument("--model_type", type = str, default = "lstm", choices = ['rnn', 'lstm', 'gru'], help = "Model architecture")
    parser.add_argument("--data_size", type = int, default = 10000, help = "Size of synthetic dataset")

    return parser.parse_args()

def main():
    logging.basicConfig(
        level = logging.INFO, 
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        handlers = [logging.StreamHandler()]
    )

    # Parse arguments and configure environment
    args = args_parser()
    setup_seed(args.seed)
    device = get_device()

    base_dir = "chapter_06_rnn_lstm_seq"
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    res_dir = os.path.join(base_dir, "results")
    os.makedirs(ckpt_dir, exist_ok = True)
    os.makedirs(res_dir, exist_ok = True)

    # Initialize dataset
    logger.info("Initializing dataset...")
    dataset = SyntheticNameDataset(num_samples = args.data_size)
    dataset.save_vocab(os.path.join(base_dir, "data/vocab.txt"))
    dataset.save_data(os.path.join(base_dir, "data/synthetic_names.txt"))

    # Split Train/Val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Collator with vocab access
    collator = VectorizedCollator(dataset.vocab)

    train_loader = DataLoader(
        train_set, 
        batch_size = args.batch_size, 
        shuffle = True, 
        collate_fn = collator,
        num_workers = 0 # Avoid MPS multiprocessing issues
    )

    logger.info(f"Vocab Size: {len(dataset.vocab)} | Classes: {len(dataset.classes)}")

    model = DynamicRNNClassifier(
        vocab_size = len(dataset.vocab),
        embedding_dim = args.embed_dim,
        hidden_dim = args.hidden_dim,
        output_dim = len(dataset.classes),
        num_layers = 2,
        bidirectional = True,
        dropout = 0.3,
        model_type = args.model_type
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    # 7. Training Loop
    metrics = {'loss': [], 'acc': []}
    
    with Timer() as t:
        for epoch in range(args.epochs):
            avg_loss, avg_acc = train_one_epoch(
                model, 
                train_loader, 
                criterion, 
                optimizer, 
                device, 
                epoch + 1
            )
            
            metrics['loss'].append(avg_loss)
            metrics['acc'].append(avg_acc)
            
            logger.info(f"Epoch {epoch + 1}/{args.epochs} - Loss: {avg_loss:.4f} - Acc: {avg_acc:.4f}")

    # 8. Save Artifacts
    model_path = os.path.join(ckpt_dir, f"{args.model_type}_model.pth")
    log_model_info(model)
    torch.save(model.state_dict(), model_path)
    save_json(metrics, os.path.join(res_dir, f"{args.model_type}_metrics.json"))
    logger.info(f"Model saved to {model_path}")

    # 9. Final Inference Test
    test_samples = ["petrov", "jackson", "rossini", "yamamoto", "papadopoulos", "unknownstuff"]
    evaluate_inference(model, dataset.vocab, dataset.classes, device, test_samples)

if __name__ == '__main__':
    main()
    
