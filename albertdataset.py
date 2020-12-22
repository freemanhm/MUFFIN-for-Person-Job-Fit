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
# load the data
# token 标识resume 或者job
def load_csv(config):
    '''
    load the csv data df_user and df_job
    '''
    df_user = pd.read_csv(config.user_processed_path,index_col=0)
    df_job = pd.read_csv(config.job_processed_path,index_col=0)
    user_cols = ['user_id'] + config.user_co_cols + config.user_ca_num_cols + \
                config.user_ca_text_cols + config.user_exp_col
    job_cols = ['jd_no'] + config.job_co_cols + config.job_ca_num_cols + \
               config.job_ca_text_cols + config.job_jd_col
    df_user = df_user[user_cols]
    df_job = df_job[job_cols]
    return df_user,df_job,user_cols,job_cols

def load_dicts(feature2idx_path):
    '''
    load the feature2idx dictionary
    :param feature2idx_path: 路径
    :return: dictionary
    '''
    with open(feature2idx_path,'r') as file:
        feature2idx = eval(file.read())
    return feature2idx


class AlBertDataset(Dataset):
    def __init__(self, df_user, df_job, user_cols, job_cols, dicts, config, token):
        super(AlBertDataset,self).__init__()
        self.df_user = df_user
        self.df_job = df_job
        self.config = config
        self.df_label = self.load_label(token)
        self.job_cols = job_cols
        self.user_cols = user_cols
        self.dicts = dicts
    
    def load_label(self,token):
        pairs = []
        labels = []
        assert token in ['train','valid','test']
        filepath = os.path.join(self.config.splitdata_path,'data.{}'.format(token))
        time_print('\nloading from {}'.format(filepath))
        sys.stdout.flush()
        df_label = pd.read_csv(filepath,sep='\t')
        df_label.columns = ['user_id','jd_no','label']
        return df_label

    def getx(self,user_id,job_id):
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
    def get_co_cols_idx(self,x):
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

    def get_city_idx(self, x):
        '''
        :param dicts: preprocess中datapre保存下来的类别字典
        :param x: 单个user和job的特征信息合并
        :return: 返回city的xi,xv
        '''
        city_xi = []
        city_xv = []
        for i in range(11, 16):
            city_xi.append(self.gen('city', str(x[i])))
            city_xv = [1] * len(city_xi)
        return city_xi, city_xv

    # 对industry,jd_type,jd_title，用于ALBERT做Embedding
    def get_cat_tokenizer(self, tokenizer, x):
        cat_token_tensors = []
        for i in range(16, 25):
            text = re.sub('/', '', x[i])
            sen_code = tokenizer.encode_plus(text, padding='max_length', truncation=True,
                                             max_length=config.cat_token_len)
            cat_token_tensors.append(sen_code.input_ids)
        return cat_token_tensors

    # 对experience，job_description，用于ALBERT做Embedding
    def get_exp_tokenizer(self, tokenizer, x):
        for i in range(len(x) - 2, len(x) - 1):
            text = re.sub(r'[\|,\\/]', '', x[i])
            sen_code = tokenizer.encode_plus(text, padding='max_length', truncation=True,
                                             max_length=config.experience_token_len)
        return sen_code.input_ids

    def get_jd_tokenizer(self, tokenizer, x):
        '''
        对job_description文本tokenizer
        :param tokenizer:
        :param x:
        :return:
        '''
        for i in range(len(x) - 1, len(x)):
            text = re.sub(r'[\|,\\/]', '', x[i])
            sen_code = tokenizer.encode_plus(text, padding='max_length', truncation=True,
                                             max_length=config.jd_token_len)
        return sen_code.input_ids

    def get_idx(self,x):
        tokenizer = BertTokenizer.from_pretrained(self.config.albertpath)
        co_xi, co_xv = self.get_co_cols_idx(x)
        city_xi, city_xv = self.get_city_idx(x)
        cat_token_tensors = self.get_cat_tokenizer(tokenizer, x)
        exp_tensor = self.get_exp_tokenizer(tokenizer, x)
        jd_tensor = self.get_jd_tokenizer(tokenizer, x)
        # 将list转化为tensor
        co_xi = torch.tensor(co_xi)
        co_xv = torch.tensor(co_xv)
        city_xi = torch.tensor(city_xi)
        city_xv = torch.tensor(city_xv)
        # 这里有多个特征，将其concat成一维向量（8*20）
        cat_token_tensors = torch.tensor(cat_token_tensors)# cat_num*token_max_len
        cat_token_tensors = cat_token_tensors.view(-1)
        exp_tensor = torch.tensor(exp_tensor)
        jd_tensor = torch.tensor(jd_tensor)

        return co_xi, co_xv, city_xi, city_xv, cat_token_tensors, exp_tensor, jd_tensor

    def __getitem__(self, index):
        user_id = self.df_label.iloc[index]['user_id']
        job_id = self.df_label.iloc[index]['jd_no']
        label = torch.FloatTensor([int(self.df_label.iloc[index]['label'])])
        # x是 user 和 job 的特征属性拼接
        x = self.getx(user_id,job_id)
        co_xi, co_xv, city_xi, city_xv, cat_token_tensors, exp_tensor, jd_tensor = self.get_idx(x)

        return user_id,job_id,co_xi, co_xv, city_xi, city_xv, cat_token_tensors, exp_tensor, jd_tensor,label
    
    def __len__(self):
        return self.df_label.shape[0]