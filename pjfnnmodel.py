# PJFNN model
import numpy as np
import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, channels, kernel_size, pool_size, dim, method='max'):
        super(TextCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[0]),
            nn.BatchNorm2d(channels),  # 其中的参数是通道数
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[1]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, dim))  # （1，dim）是指输出大小
        )

        if method is 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, dim))
        elif method is 'mean':
            self.pool = nn.AdaptiveAvgPool2d((1, dim))
        else:
            raise ValueError('method {} not exist'.format(method))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x).squeeze(2)
        x = self.pool(x).squeeze(1)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class PJFNN(nn.Module):
    def __init__(self,config):
        super(PJFNN, self).__init__()
        self.config = config
        self.emb = nn.Embedding(self.get_words_count(), config.dim, padding_idx=0)
        self.resume_part = TextCNN(
            channels=config.sent_num['resume'],
            kernel_size=[(5, 1), (3, 1)],
            pool_size=(2, 1),
            dim=config.dim,
            method='max'
        )

        self.job_part = TextCNN(
            channels=config.sent_num['job'],
            kernel_size=[(5, 1), (5, 1)],
            pool_size=(2, 1),
            dim=config.dim,
            method='mean'
        )

        self.mlp = MLP(
            input_size=config.dim,
            output_size=1
        )

    def get_words_count(self):
        '''
        read the dict of idx2word
        '''
        with open(self.config.idx2word_path,'r') as file:
            word2idx = eval(file.read())
        return len(word2idx)

    def forward(self, resume_sent, job_sent):
        resume_vec, job_vec = self.emb(resume_sent), self.emb(job_sent)
        resume_vec, job_vec = self.resume_part(resume_vec), self.job_part(job_vec)
        x = resume_vec * job_vec
        x = self.mlp(x).squeeze(1)
        return x
