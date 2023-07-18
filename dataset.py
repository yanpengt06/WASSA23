import argparse
import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import json
import numpy as np
import random
from pandas import DataFrame
from transformers import AutoTokenizer

"""
    dataset.py: define your Dataset class here.
"""


class SST2(Dataset):
    def __init__(self, dataset_name='sst2', split="train"):
        dataset = self.read(dataset_name)[split]  # ('sentence', 'label', 'att_mask', 'input_ids', xxx)
        # (input_ids, attention_mask, token_type_ids, label, sentence)
        self.data = list(zip(dataset['input_ids'], dataset['attention_mask'], dataset['token_type_ids'],
                             dataset['label'], dataset['sentence']))

    def read(self, dataset_name):
        raw_dataset = load_dataset("glue", dataset_name)  # ('sentence', 'label', 'idx')
        ckpt = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(ckpt)

        def tokenize_fun(examples):
            return tokenizer(examples['sentence'], truncation=True)

        tokenized_dataset = raw_dataset.map(tokenize_fun, batched=True)  # add ('input_ids', 'att_mask', xxx)
        return tokenized_dataset

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, datas):
        """
        @params:
        datas: a list of data: (input_ids, att_mask, token_type_ids, label, sentence)
        """
        # input_ids, pad_token_id of bert-uncased is 0.
        input_ids = (pad_sequence([torch.LongTensor(d[0]) for d in datas], padding_value=0, batch_first=True))  # B x M
        # padding att_mask is 0
        att_mask = (pad_sequence([torch.LongTensor(d[1]) for d in datas], padding_value=0, batch_first=True))  # B x M
        token_type_ids = (
            pad_sequence([torch.LongTensor(d[2]) for d in datas], padding_value=1, batch_first=True))  # B x M
        label = torch.LongTensor([d[3] for d in datas])  # B
        sentence = [d[4] for d in datas]  # B, a list of strings
        return input_ids, att_mask, token_type_ids, label, sentence


class IEMOCAPDataset(Dataset):

    def __init__(self, dataset_name='IEMOCAP', split='train', speaker_vocab=None, label_vocab=None, args=None,
                 tokenizer=None):
        self.speaker_vocab = speaker_vocab
        self.label_vocab = label_vocab
        self.args = args
        self.data = self.read(dataset_name, split, tokenizer)
        print(len(self.data))

        self.len = len(self.data)

    def read(self, dataset_name, split, tokenizer):
        with open('./data/%s/%s_data_roberta_v2.json.feature' % (dataset_name, split), encoding='utf-8') as f:
            raw_data = json.load(f)

        # process dialogue
        dialogs = []
        # raw_data = sorted(raw_data, key=lambda x:len(x))
        for d in raw_data:
            # if len(d) < 5 or len(d) > 6:
            #     continue
            utterances = []
            labels = []
            speakers = []
            features = []
            for i, u in enumerate(d):
                utterances.append(u['text'])
                labels.append(self.label_vocab['stoi'][u['label']] if 'label' in u.keys() else -1)
                speakers.append(self.speaker_vocab['stoi'][u['speaker']])
                features.append(u['cls'])
            dialogs.append({
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'features': features
            })
        random.shuffle(dialogs)
        return dialogs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''
        return torch.FloatTensor(self.data[index]['features']), \
               torch.LongTensor(self.data[index]['labels']), \
               self.data[index]['speakers'], \
               len(self.data[index]['labels']), \
               self.data[index]['utterances']

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        '''
        :param data:
            features, labels, speakers, length, utterances
        :return:
            features: (B, N, D) padded
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[3] for d in data])
        feaures = pad_sequence([d[0] for d in data], batch_first=True)  # (B, N, D)
        labels = pad_sequence([d[1] for d in data], batch_first=True, padding_value=-1)  # (B, N )
        # adj = self.get_adj_v1([d[2] for d in data], max_dialog_len)
        # s_mask, s_mask_onehot = self.get_s_mask([d[2] for d in data], max_dialog_len)
        lengths = torch.LongTensor([d[3] for d in data])
        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)  # B x N
        utterances = [d[4] for d in data]

        return feaures, labels, speakers, lengths, utterances


class WASSASingle(Dataset):

    def __init__(self, split='tr', args=None, tokenizer=None):
        self.args = args
        self.data = self.read(split)  # (text, pola, itst, emp)
        self.tokenizer = tokenizer
        self.len = len(self.data)
        print(f'This dataset has {self.len} examples.')

    def read(self, split):
        with open(f'./single_text_{split}.txt', 'r', encoding='utf-8') as f:
            texts = f.readlines()
        polas = []
        itsts = []
        emps = []
        with open(f'./single_labels_{split}.txt', 'r') as f2:
            lines = f2.readlines()
            labels = [line.strip().split(' ') for line in lines]
            for line in labels:
                polas.append(float(line[0]))
                itsts.append(float(line[1]))
                emps.append(float(line[2]))
        return list(zip(texts, polas, itsts, emps))

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            (text,
            polarity,
            intensity,
            empathy)
        '''
        return self.data[index]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        '''
        :param data:
            a batch of data, each is formatted as (text, pola, itst, emp)
        :return:

        '''
        encoded = self.tokenizer([d[0] for d in data], padding=True, truncation=True, return_tensors='pt')  # texts
        polas = torch.FloatTensor([d[1] for d in data])  # B x 1
        itsts = torch.FloatTensor([d[2] for d in data])  # B x 1
        emps = torch.FloatTensor([d[3] for d in data])  # B x 1

        # speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)  # B x N
        texts = [d[0] for d in data]

        return encoded['input_ids'], encoded['attention_mask'], polas, itsts, emps, texts


class WASSAContext(Dataset):

    def __init__(self, split='train', args=None, tokenizer=None):
        self.args = args
        self.tokenizer = tokenizer
        self.data = self.read(split, args.window_size)  # (text, pola, itst, emp)
        self.len = len(self.data)
        print(f'Loaded {split} context dataset with window size {args.window_size}')
        print(f'This dataset has {self.len} examples.')

    def read(self, split, window_size):
        if f'{split}_context_w{window_size}.pkl' in os.listdir('./data/context'):
            # read from pickle
            with open(f'./data/context/{split}_context_w{window_size}.pkl', 'rb') as f4:
                datas = pickle.load(f4)
            print(f'{split} context dataset with window size {window_size} loaded from pkl!')
            return datas
        else:
            # tokenize
            with open(f'./data/context/{split}_context_w{window_size}.txt', 'r', encoding='utf-8') as f2:
                lines = f2.readlines()
            utts = [line.strip() for line in lines]
            encoded = self.tokenizer(utts, truncation=True)
            input_ids = encoded['input_ids']
            att_mask = encoded['attention_mask']
            polas = []
            itsts = []
            emps = []
            with open(f'./data/context/{split}_context_labels.txt', 'r') as f2:
                lines = f2.readlines()
                lines = [line.strip().split(' ') for line in lines]
                for line in lines:
                    polas.append(float(line[0]))
                    itsts.append(float(line[1]))
                    emps.append(float(line[2]))
            datas = list(zip(utts, polas, itsts, emps, input_ids, att_mask))
            with open(f'./data/context/{split}_context_w{window_size}.pkl', 'wb') as f3:
                pickle.dump(datas, f3)
            return datas

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            (text,
            polarity,
            intensity,
            empathy,
            input_ids,
            att_mask)
        '''
        return self.data[index]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        '''
        :param data:
            a batch of data, each is formatted as (text, pola, itst, emp, input_ids, att_mask)
        :return:

        '''
        input_ids = pad_sequence([torch.LongTensor(d[-2]) for d in data], batch_first=True, padding_value=1, )
        att_mask = pad_sequence([torch.LongTensor(d[-1]) for d in data], batch_first=True)  # B x M
        polas = torch.FloatTensor([d[1] for d in data])  # B
        itsts = torch.FloatTensor([d[2] for d in data])  # B
        emps = torch.FloatTensor([d[3] for d in data])  # B

        # speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)  # B x N
        texts = [d[0] for d in data]

        return input_ids, att_mask, polas, itsts, emps, texts


class WASSATest(Dataset):

    def __init__(self, path):
        self.data = self.read(path)  # dev or test dataset { 'text', 'attention_mask', 'input_ids', 'label', 'token_type_ids'}
        # no 'label' for final test!
        self.len = len(self.data)
        print(f'This dataset has {self.len} datas.')

    def read(self, path):
        with open(path, 'rb') as f2:
            dev_dataset = pickle.load(f2)
        datas = list(
            zip(dev_dataset['text'], dev_dataset['input_ids'], dev_dataset['token_type_ids'], \
                dev_dataset['attention_mask']))
        return datas

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            (text,
            input_ids,
            token_type_ids,
            att_mask)
        '''
        return self.data[index]

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        '''
        :param data:
            a batch of data, each is formatted as (text, pola, itst, emp, input_ids, att_mask)
        :return:

        '''
        input_ids = pad_sequence([torch.LongTensor(d[1]) for d in data], batch_first=True)
        att_mask = pad_sequence([torch.LongTensor(d[-1]) for d in data], batch_first=True)  # B x M
        token_type_ids = pad_sequence([torch.LongTensor(d[-2]) for d in data], batch_first=True)  # B x M
        # speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first=True, padding_value=-1)  # B x N
        texts = [d[0] for d in data]

        return input_ids, att_mask, token_type_ids, texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.window_size = -1
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    train_set = WASSAContext('train', args=args, tokenizer=tokenizer)
    train_loader = DataLoader(train_set, batch_size=8, collate_fn=train_set.collate_fn)
    for idx, batch in enumerate(train_loader):
        input_ids = batch[0]
        att = batch[1]
        polas = batch[2]
        texts = batch[-1]
        print(input_ids.shape)
        print(polas.shape)
        print(texts)
        if idx == 20:
            break
