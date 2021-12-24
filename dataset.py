import os
import re

import nltk
import pandas as pd
import torch
from nltk import sent_tokenize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer

nltk.download('punkt')


class MauiDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)
        self.sent = self.dataset.sentence
        self.y = self.dataset.labels
        self.source_len = 256
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tag2idx = {'1': 1, '2': 2, '3': 3}

    def __len__(self):
        return len(self.sent)

    def __getitem__(self, index):
        sent_ = str(self.sent[index])
        sent_splitted = ' '.join(sent_.split())
        encod = self.tokenizer.batch_encode_plus([sent_splitted], max_length=self.source_len,
                                                 pad_to_max_length=True, return_tensors='pt', add_special_tokens=False)

        label_list = list(map(self.tag2idx.get, self.y.apply(lambda x: x.strip('[]').split(', '))[index]))
        labels = torch.tensor(label_list + [0] * (self.source_len - len(label_list)) if len(label_list) < self.source_len
                              else label_list[:self.source_len], dtype=torch.float)

        sent_ids = encod['input_ids'].squeeze()

        return {
            'sent_ids': sent_ids.to(dtype=torch.long),
            'target': labels
        }


def convert(key, train_path, filekey):
    sentences = ""
    for line in open(train_path + "/" + filekey[key], 'r'):
        sentences += (" " + line.rstrip())
    tokens = sent_tokenize(sentences)
    key_file = open(train_path + "/" + str(key), 'r')
    keys = [line.strip() for line in key_file]
    key_sent = []
    labels = []
    for token in tokens:
        z = ['O'] * len(token.split())
        for k in keys:
            if k in token:

                if len(k.split()) == 1:
                    try:
                        z[token.lower().split().index(k.lower().split()[0])] = 'B'
                    except ValueError:
                        continue
                elif len(k.split()) > 1:
                    try:
                        if token.lower().split().index(k.lower().split()[0]) and token.lower().split().index(
                                k.lower().split()[-1]):
                            z[token.lower().split().index(k.lower().split()[0])] = 'B'
                            for j in range(1, len(k.split())):
                                z[token.lower().split().index(k.lower().split()[j])] = 'I'
                    except ValueError:
                        continue
        for m, n in enumerate(z):
            if z[m] == 'I' and z[m - 1] == 'O':
                z[m] = 'O'

        if set(z) != {'O'}:
            labels.append(z)
            key_sent.append(token)
    return key_sent, labels


def prepare_data():
    train_path = "data/maui-semeval2010-train"
    txt = sorted([f for f in os.listdir(train_path) if
                  not f.endswith("-justTitle.txt") and not f.endswith(".key") and not f.endswith("-CrowdCountskey")])
    key = sorted([f for f in os.listdir(train_path) if f.endswith(".key")])
    train_df = pd.DataFrame(columns=['sentence', 'labels'])
    test_df = pd.DataFrame(columns=['sentence', 'labels'])
    tag2idx = {'B': 1, 'I': 2, 'O': 3}
    filekey = dict()
    for i, k in enumerate(txt):
        filekey[key[i]] = k
    sentences_ = []
    labels_ = []
    for key, value in filekey.items():
        s, l = convert(key, train_path, filekey)
        sentences_.append(s)
        labels_.append(l)
    sentences = [item for sublist in sentences_ for item in sublist]
    labels = [item for sublist in labels_ for item in sublist]
    labels_mapped = [list(map(tag2idx.get, i)) for i in labels]
    train_df['sentence'] = sentences[:round(len(sentences) * 0.8)]
    test_df['sentence'] = sentences[round(len(sentences) * 0.8):]
    train_df['labels'] = labels_mapped[:round(len(labels_mapped) * 0.8)]
    test_df['labels'] = labels_mapped[round(len(labels_mapped) * 0.8):]
    train_df['tokens'] = train_df['sentence'].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x).lower().split())\
        .astype(object)
    test_df['tokens'] = test_df['sentence'].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x).lower().split()).astype(
        object)
    train_df['labels'] = tokenize_and_align_labels(train_df)
    test_df['labels'] = tokenize_and_align_labels(test_df)

    train_df.to_csv('data/train.csv')
    test_df.to_csv('data/test.csv')


def prepare_data_for_training():
    train_data = MauiDataset('data/train.csv')
    test_data = MauiDataset('data/test.csv')
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4)
    return train_loader, test_loader


def tokenize_and_align_labels(example):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenized_inputs = tokenizer(list(example["tokens"]), truncation=True, is_split_into_words=True)
    label_all_tokens = True
    labels = []
    for i, label in enumerate(list(example['labels'])):
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return labels


if __name__ == '__main__':
    prepare_data()