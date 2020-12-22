#!/usr/bin/env python
# coding: utf-8

import sys, os, random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from config import DefaultConfig
from torch.utils.data import DataLoader
from utils.utils import *
import pandas as pd
import numpy as np
from transformers import AlbertConfig, BertTokenizer, AlbertModel
import re

config = DefaultConfig()

def load_sents(token):
    sents = {}
    sentnum = {}  # 用于标记已有句子数目
    tensor_size = [config.sent_num[token], config.word_num[token]]  # 每个resume or job 张量大小，维度
    filepath = os.path.join(config.train_path, '{}.sent.id'.format(token))
    time_print('\nloading from {}'.format(filepath))
    sys.stdout.flush()
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            idx, sent = line.strip().split('\t')
            if idx not in sents:
                sents[idx] = torch.zeros(tensor_size).long()
                sentnum[idx] = 0
            if sentnum[idx] == config.sent_num[token]: continue  # 达到最大句子数，跳转继续读文件中的下一行
            sent = torch.LongTensor([int(x) for x in sent.split(' ')])
            sents[idx][sentnum[idx]] = F.pad(sent, (0, config.word_num[token] - len(sent)))
            sentnum[idx] += 1
    return sents

# load the data
# token 标识resume 或者job
def load_csv(config):
    '''
    load the csv data df_user and df_job
    '''
    df_user = pd.read_csv(config.user_processed_path, index_col=0)
    df_job = pd.read_csv(config.job_processed_path, index_col=0)
    user_cols = ['user_id'] + config.user_co_cols + config.user_ca_num_cols + \
                config.user_ca_text_cols + config.user_exp_col
    job_cols = ['jd_no'] + config.job_co_cols + config.job_ca_num_cols + \
               config.job_ca_text_cols + config.job_jd_col
    df_user = df_user[user_cols]
    df_job = df_job[job_cols]
    return df_user, df_job, user_cols, job_cols


def load_dicts(feature2idx_path):
    '''
    load the feature2idx dictionary
    :param feature2idx_path: 路径
    :return: dictionary
    '''
    with open(feature2idx_path, 'r') as file:
        feature2idx = eval(file.read())
    return feature2idx


class PJFFFDataset(Dataset):
    def __init__(self, df_user, df_job, user_cols, job_cols, user_sents, job_sents, dicts, config, token):
        super(PJFFFDataset, self).__init__()
        self.df_user = df_user
        self.df_job = df_job
        self.user_sents = user_sents #load_sents('resume')
        self.job_sents = job_sents #load_sents('job')
        self.config = config
        self.df_label = self.load_label(token)
        self.job_cols = job_cols
        self.user_cols = user_cols
        self.dicts = dicts

    def load_label(self, token):
        pairs = []
        labels = []
        assert token in ['train', 'valid', 'test']
        filepath = os.path.join(self.config.splitdata_path, 'data.{}'.format(token))
        time_print('\nloading from {}'.format(filepath))
        sys.stdout.flush()
        df_label = pd.read_csv(filepath, sep='\t')
        df_label.columns = ['user_id', 'jd_no', 'label']
        return df_label

    def getx(self, user_id, job_id):
        '''
        根据两个id拼接user和job 的特征属性
        :param user_id:
        :param job_id:
        :return: np.array
        '''
        user = self.df_user[self.df_user['user_id'] == user_id].values[0]
        job = self.df_job[self.df_job['jd_no'] == job_id].values[0]
        data = np.hstack((user, job)).reshape(-1, 1)
        df = pd.DataFrame(data=data.T, columns=self.user_cols + self.job_cols)
        df = df[config.cols]
        mergedata = df.values[0]
        return mergedata

    # 对于co_cols,生成xi, xv，用于Embedding层
    def get_co_cols_idx(self, x):
        '''
        xv已在数据预处理阶段归一化
        :param x:
        :return:
        '''
        co_xi = []
        co_xv = list(x[0:11])
        for i in range(11):
            if i < 3:
                co_xi.append(i + 1)
            elif i < 5:
                co_xi.append(4)
            elif i < 7:
                co_xi.append(5)
            else:
                co_xi.append(6)
        return co_xi, co_xv

    # 对于city类别，用于Embedding层(类别型）
    def gen(self, idx, key):
        '''
        :param dicts: preprocess中datapre保存下来的类别字典
        :param idx: 这里只用到'city'
        :param key: 就是原始值
        :return:对应的num
        '''
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def get_cat_idx(self, x):
        """
        :param dicts: preprocess中datapre保存下来的类别字典
        :param x: 单个user和job的特征信息合并
        :return: 返回city,industry,type的xi,xv
        注意其中的x[20]为job_title，其编码全为零，在model训练中不能使用
        """
        cat_xi = []
        cat_xv = []
        for i in range(11, 16):
            cat_xi.append(self.gen('city', str(x[i])))
        for i in range(16, 21):
            cat_xi.append(self.gen('industry', str(x[i])))
        for i in range(21, 25):
            cat_xi.append(self.gen('type', str(x[i])))
        cat_xv = [1] * len(cat_xi)
        return cat_xi, cat_xv


    def get_idx(self, x):
        co_xi, co_xv = self.get_co_cols_idx(x)
        cat_xi, cat_xv = self.get_cat_idx(x)
        # 将list转化为tensor
        co_xi = torch.tensor(co_xi)
        co_xv = torch.tensor(co_xv)
        cat_xi = torch.tensor(cat_xi)
        cat_xv = torch.tensor(cat_xv)

        return co_xi, co_xv, cat_xi, cat_xv

    def __getitem__(self, index):
        user_id = self.df_label.iloc[index]['user_id']
        job_id = self.df_label.iloc[index]['jd_no']
        label = torch.FloatTensor([int(self.df_label.iloc[index]['label'])])
        # x是 user 和 job 的特征属性拼接
        x = self.getx(user_id, job_id)
        co_xi, co_xv, cat_xi, cat_xv = self.get_idx(x)
        # 获取user和job的文本分词编码
        exp_sent = self.user_sents[user_id]
        jd_sent = self.job_sents[job_id]

        return user_id, co_xi, co_xv, cat_xi, cat_xv, exp_sent, jd_sent, label

    def __len__(self):
        return self.df_label.shape[0]