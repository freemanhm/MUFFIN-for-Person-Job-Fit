# coding: utf-8

import sys, os, random
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.utils import *
import pandas as pd
from config import DefaultConfig
config = DefaultConfig()


# load the data
# token 标识resume 或者job
def load_csv(config):
    '''
    load the csv data that the features have been mapped into num
    '''
    df_user = pd.read_csv(config.user_num_path, index_col=0)
    df_job = pd.read_csv(config.job_num_path, index_col=0)
    return df_user, df_job


class deepFMDataset(Dataset):
    def __init__(self, df_user, df_job, config, token):
        # token用于标注从train，valid，test文件读
        super(deepFMDataset, self).__init__()
        self.df_user = df_user
        self.df_job = df_job
        self.config = config
        self.cols = config.fmco_cols + config.fmca_cols
        self.df_label = self.load_label(token)
        self.co_index = self.continuousindex()
        self.df_idx, self.df_value = self.convert_features()


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

    def continuousindex(self):
        # continuous features and category features
        co_cols = self.config.fmco_cols
        #         ca_cols = self.config.user_ca_cols + self.config.job_ca_cols
        cnt = 1
        co_index = {}
        for col in co_cols:
            co_index[col] = cnt  # 列名映射为索引
            cnt += 1
        return co_index

    def convert_features(self):
        '''
        concat the user and job with the pairs and labels
        convert the continuous features into index
        generate the xv for category features
        '''
        # 拼接df_user,df_job pairs,labels
        df = pd.merge(self.df_label, self.df_user, on=['user_id'], how='left')
        df = pd.merge(df, self.df_job, on=['jd_no'], how='left')
        # set the features order
        cols = ['user_id', 'jd_no', 'label'] + self.cols
        df = df[cols]
        # 数据index 对应产出xi
        df_idx = df.copy(deep=True)
        # convert continuous features into index
        co_cols = self.config.fmco_cols
        num = df_idx.shape[0]
        for col in co_cols:
            df_idx[col] = pd.Series([self.co_index[col]] * num)
        # 数据value 对应产出xv
        ca_cols = self.config.fmca_cols
        for col in ca_cols:
            df[col] = pd.Series([1] * num)
        return df_idx, df


    def __getitem__(self, index):
        '''
        get a sample data based on the index
        the data included each feature's index xi and feature value xv
        consider the continuous features and category features respectively
        '''
        # continuous features value convertered into inex xi
        user = self.df_value.iloc[index]['user_id']
        label = torch.FloatTensor([int(self.df_value.iloc[index]['label'])])
        #         label = torch.tensor(self.df_value.iloc[index][['label']].astype(int).values)
        xi = torch.LongTensor(self.df_idx.iloc[index][self.cols].astype(int).values)
        xv = torch.LongTensor(self.df_value.iloc[index][self.cols].astype(int).values)
        return user, xi, xv, label

    def __len__(self):
        return self.df_label.shape[0]


if __name__ == '__main__':

    df_user, df_job = load_csv(config)

    train_dataset = deepFMDataset(df_user, df_job, config, 'train')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size['train'],
        shuffle=True,
        num_workers=config.num_workers['train']
    )
    i = 0
    for i, (user, xi, xv, label) in enumerate(train_loader):
        if i < 1:
            #             print(user)
            #             user = list(user)
            #             print(user)
            #             print(xi)
            #             print(label)
            print(xi)
            print(xv[0:2, 0:10])
            print(len(label))
            i += 1
        else:
            break

