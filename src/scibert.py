# Standard library
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

# Third-party
import numpy as np
import pandas as pd
from termcolor import colored
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, get_cosine_schedule_with_warmup, AutoTokenizer

from base_predictor import BasePredictor, LABELS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


IGNORE_LABEL = -100


def tokenize_and_preserve_labels(tokens, labels, tokenizer):
    tokenized_sentence = []
    labels = []
    for token, label in zip(tokens, labels):
        tokenized_word = tokenizer.tokenize(token)
        n_subwords = len(tokenized_word)
        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


class dataset(Dataset):
    def __init__(self, all_tokens, all_labels, tokenizer, max_len):
        self.all_tokens = all_tokens
        self.all_labels = all_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Build word-level dataset
        label_set = []
        for label in LABELS:
            label_set.append(f'B-{label}')
            label_set.append(f'I-{label}')
        label_set.append('O')

        self.id2label = {i: l for i, l in enumerate(label_set)}
        self.id2label['O'] = IGNORE_LABEL
        self.label2id = {l: i for i, l in self.id2label.items()}

    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        tokens = self.all_tokens[index]
        labels = self.all_labels[index]
        tokenized_sentence, labels = tokenize_and_preserve_labels(tokens, labels, self.tokenizer)

        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = ["<s> "] + tokenized_sentence + [" </s>"] # add special tokens of Roberta
        labels.insert(0, "O") # add outside label for [CLS] token
        labels.append("O") # add outside label for [SEP] token

        # step 3: truncating/padding
        maxlen = self.max_len

        if (len(tokenized_sentence) > maxlen):
          # truncate
          tokenized_sentence = tokenized_sentence[:maxlen]
          labels = labels[:maxlen]
        else:
          # pad
          tokenized_sentence = tokenized_sentence + ['<pad>'for _ in range(maxlen - len(tokenized_sentence))]
          labels = labels + ["O" for _ in range(maxlen - len(labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '<pad>' else 0 for tok in tokenized_sentence] #modifié selon https://huggingface.co/docs/transformers/v4.21.1/en/model_doc/camembert

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        label_ids = [self.label2id[label] for label in labels]
        # the following line is deprecated
        #label_ids = [label if label != 0 else -100 for label in label_ids]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return len(self.all_tokens)


def build_dataset(folder: Path, tokenizer: Any):
    all_tokens = []
    all_labels = []
    file_paths = sorted([x for x in folder.rglob('*.tsv')])

    for file in file_paths:
        tokens = []
        labels = []
        for line in file.read_text().split('\n'):
            line = line.strip()
            if line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 5:
                continue

            _, _, token, _, label = parts
            label = 'O' if label == '_' else f'I-{label}'
            label = label.split('[')[0]
            tokens.append(token)
            labels.append(label)

        for idx in range(len(labels)-1):
            if labels[idx] == 'O' and labels[idx+1].startswith('I-'):
                labels[idx+1] = labels[idx+1].replace('I-', 'B-')

        all_tokens.append(tokens)
        all_labels.append(labels)

    # data = {"tokens": all_tokens, "ner_tags": all_labels, 'id2label': id2label, 'label2id': label2id}
    # tokenized_sentence = []
    # labels = []
    # for tokens, labels in zip(all_tokens, all_labels):
    #     for token, label in zip(tokens, labels):
    #         tokenized_word = tokenizer.tokenize(token)
    #         n_subwords = len(tokenized_word)
    #         # Add the tokenized word to the final tokenized word list
    #         tokenized_sentence.extend(tokenized_word)

    #         # Add the same label to the new list of labels `n_subwords` times
    #         labels.extend([label] * n_subwords)

    # import ipdb; ipdb.set_trace()
    ds = dataset(all_tokens, all_labels, tokenizer, max_len=256)
    return ds


def main(args):
    MAX_LEN = 256
    train_folder = Path("./data/train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, from_tf=False, model_max_length=MAX_LEN)
    ds = build_dataset(train_folder, tokenizer)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    main(args)


# if __name__ == "__main__":
#     paths = Path(".").glob("*.conll")

#     ap = argparse.ArgumentParser()
#     ap.add_argument("--train", type=Path, required=False)
#     ap.add_argument("--valid", type=Path)
#     ap.add_argument("--test", type=Path)

#     ap.add_argument("--output_dir", type=Path, required=False)
#     ap.add_argument("--epochs", type=int, default=3)
#     ap.add_argument("--batch_size", type=int, default=8)
#     ap.add_argument("--lr", type=float, default=5e-5)
#     ap.add_argument("--max_length", type=int, default=256)
#     args = ap.parse_args()

    # TRAIN_BATCH_SIZE = 16
    # VALID_BATCH_SIZE = 16
    # EPOCHS = 5#3#20
    # LEARNING_RATE = 5e-5 #1e-05
    # MAX_GRAD_NORM = 10
    # MAX_LEN = 256


    # phase = 'train'
    # base_path = Path(f'data/{phase}')
    # file_paths = sorted([x for x in base_path.rglob('*.tsv')])
    # data = build_dataset(paths, tokenizer)
    # import ipdb; ipdb.set_trace()

    # # model_name = args.model

    # # dataset = build_dataset(paths)
    # # import ipdb; ipdb.set_trace()

    # label_set = []
    # for label in LABELS:
    #     label_set.append(f'B-{label}')
    #     label_set.append(f'I-{label}')
    # label_set.append('O')

    # id2label = {i: l for i, l in enumerate(label_set)}
    # id2label['O'] = IGNORE_LABEL
    # label2id = {l: i for i, l in id2label.items()}

    # # Convert string labels to ids when present during alignment

    # model = AutoModelForTokenClassification.from_pretrained(
    #     model_name,
    #     num_labels=len(label_set),
    #     id2label=id2label,
    #     label2id=label2id,
    # )

    # data_collator = DataCollatorForTokenClassification(tokenizer)


    # training_args = TrainingArguments(
    #     output_dir=str(args.output_dir),
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=args.lr,
    #     per_device_train_batch_size=args.batch_size,
    #     per_device_eval_batch_size=args.batch_size,
    #     num_train_epochs=args.epochs,
    #     weight_decay=0.01,
    #     logging_steps=50,
    #     report_to=[], # disable wandb by default
    #     load_best_model_at_end=True,
    #     metric_for_best_model="f1",
    #     greater_is_better=True,
    #     seed=42,
    # )
