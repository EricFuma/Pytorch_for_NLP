'''
Author: your name
Date: 2020-12-14 15:33:21
LastEditTime: 2020-12-15 01:07:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /fgy/classifier.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# Mine
from split_text import str_to_char
from make_vocab import make_vocab

class expDataset(Dataset):
    def __init__(self, file_path, max_sent_len, max_vocab_size=8000):
        super(expDataset, self).__init__()
        # self.sents = torch.LongTensor(np.random.randint(1, 999, (1000, 30)))
        # self.labels = torch.LongTensor(np.random.randint(0, 9, (1000,)))
        self.max_vocab_size = max_vocab_size
        sent_tensor_list, self.labels = self.get_tensorData(file_path)
        self.num_class = torch.max(self.labels).item() + 1
        sent_tensor_list = [sent_tensor[:max_sent_len] for sent_tensor in sent_tensor_list]  # truncate
        self.sents = nn.utils.rnn.pad_sequence(sent_tensor_list, batch_first=True)                # padding
        
        # print(self.sents_tensor_list[0])

    def __getitem__(self, index):
        return self.sents[index], self.labels[index]
    
    def __len__(self):
        return len(self.sents)

    def get_tensorData(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            init_sents, init_labels = [], []
            label2id = {}
            for line in lines:
                items = line.strip('_!_').split('_!_')
                label, sent = items[2], items[3]
                init_labels.append(label)
                init_sents.append(sent)
            # label 映射
            labels = []
            counter = Counter(init_labels)
            for label, num in counter.most_common():
                label2id[label] = [len(label2id), num]
            for label in init_labels:
                labels.append(label2id[label][0])
            labels = torch.LongTensor(labels)
            with open('label2id.json', 'w', encoding='utf-8') as fp:
                json.dump(label2id, fp, ensure_ascii=False, indent=4)

            # sent 切分 & 编码
            sent_list = []
            sent_tensor_list = []
            for sent in tqdm(init_sents, total=len(init_sents)):
                chars = str_to_char(sent)
                sent_list.append(chars)
            word2id = make_vocab(sent_list, self.max_vocab_size)
            for sent in sent_list:
                sent = [word2id[char] if char in word2id else 1 for char in sent]
                sent_tensor = torch.LongTensor(sent)
                sent_tensor_list.append(sent_tensor)
            print('max length of sentence: ', len(max(sent_tensor_list, key=lambda x: len(x))))
            return sent_tensor_list, labels

class Classifier(nn.Module):
    def __init__(self, num_class, max_vocab_size):
        # WHY
        super(Classifier, self).__init__()
        self.embed_layer = nn.Embedding(max_vocab_size, 128, padding_idx=0)
        self.lstm_layer = nn.LSTM(128, 256, batch_first=True, bidirectional=True)
        self.dense_layer = nn.Linear(1024, 256, bias=True)
        self.relu_func = nn.ReLU(inplace=True)
        self.classifier_layer = nn.Linear(256, num_class)
        

    def forward(self, x):
        # [b, s_l]
        embed_x = self.embed_layer(x)  # [b, s_l, embed_dim]
        _, (h, c) = self.lstm_layer(embed_x)   # [2, b, hidden_dim]
        h, c = h.reshape(h.shape[1], -1), c.reshape(c.shape[1], -1)  
        sent_reprs = torch.cat([h, c], dim=-1)  # [b, hidden_dim*4]
        output = self.dense_layer(sent_reprs)
        self.relu_func(output)
        # print(output.shape)
        # exit()
        logits = self.classifier_layer(output)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


def main():
    # Params
    USE_CUDA = True
    epochs = 5
    learning_rate = 5e-3
    batch_size = 32
    num_class = 10
    max_sent_length = 120
    max_vocab_size = 8000
    init_data_path = 'toutiao_cat_data.txt'

    # Dataset
    print("Load dataset...")
    dataset = expDataset(init_data_path, max_sent_length)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    num_class = dataset.num_class
    print("Number of class: ", num_class)

    # Model
    print("Create model...")
    model = Classifier(num_class, max_vocab_size)
    if USE_CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Logic
    print("Start training...")
    for epoch in range(epochs):
        total_losses = 0
        count = 0
        temp_loss = 0
        print('-' * 40)
        for i, (sents, labels) in tqdm(enumerate(dataloader)):
            if USE_CUDA:
                sents, labels = sents.cuda(), labels.cuda()
            # forward
            logits, probs = model(sents)
            loss = loss_fn(logits, labels)   # 因为这个loss做了log和softmax
            total_losses += loss.item()
            # backward  ->  WHY
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印中间信息
            temp_loss += torch.sum(loss).item()
            count += len(sents)
            if i % 400 == 0:
                print(total_losses / count)
        total_loss = total_losses / len(dataset)
        print("Training Information: ", "epoch-{:d}, loss-{:.4f}".format(epoch+1, total_loss.item()))
    print("Finish training!")

if __name__ == '__main__':
    main()
    # dataset = expDataset('toutiao_cat_data.txt', 120)
    # print(dataset.sents.shape)
