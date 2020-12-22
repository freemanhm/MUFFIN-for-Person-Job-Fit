# for ourmodel
import numpy as np
import torch
import torch.nn as nn
from model.albertdataset import load_csv, AlBertDataset
from torch.utils.data import DataLoader
from transformers import AlbertConfig, BertTokenizer, AlbertModel
# from config import DefaultConfig
# config = DefaultConfig()

class CoEmbdNet(nn.Module):
    '''
    continuous features Embedding向量化
    '''
    def __init__(self, config):
        super(CoEmbdNet, self).__init__()
        self.config = config
        # for the continuous features
        self.co_emb = nn.Embedding(config.co_idx+1, config.embdsize)
        nn.init.xavier_uniform_(self.co_emb.weight)

    def forward(self, xi, xv):
        # for continuous features
        co_emb = self.co_emb(xi)
        co_value = torch.mul(co_emb, xv.unsqueeze(-1))
        return co_value

class CaEmbdNet(nn.Module):
    '''
    category features  Embedding向量化 （只有city类型）
    '''
    def __init__(self, config):
        super(CaEmbdNet, self).__init__()
        self.config = config
        self.ca_idx = self.get_cat_size()
        # for the category features
        self.ca_emb = nn.Embedding(self.ca_idx+1, config.embdsize)
        nn.init.xavier_uniform_(self.ca_emb.weight)

    def get_cat_size(self):
        config = self.config
        with open(config.feature2idx_path,'r') as file:
            dicts = eval(file.read())
        return len(dicts['city'])

    def forward(self, xi, xv):
        # for continuous features
        ca_emb = self.ca_emb(xi)
        ca_value = torch.mul(ca_emb, xv.unsqueeze(-1))
        return ca_value

class AlbertEmbd(nn.Module):
    '''
       调用transformers包中的bert模型，对文本特征做Embedding
       '''

    def __init__(self, config, model_config):
        super(AlbertEmbd, self).__init__()
        self.config = config
        self.embd = AlbertModel.from_pretrained(config.albertpath, config=model_config)

    def forward(self, token_tensor,exp_tensor,jd_tensor):
        embd_all = []
        for i in range(self.config.ca_num):
            embd = self.embd(token_tensor[:, i * self.config.cat_token_len:(i + 1) * self.config.cat_token_len])
            embd_all.append(embd[1])  # 将首位CLS的向量作为句子表示输入到后续模型中
        cat_token_embd = torch.cat(embd_all, dim=-1)
        cat_token_embd = cat_token_embd.reshape(cat_token_embd.size()[0], self.config.ca_num, -1)
        exp_embd = self.embd(exp_tensor)
        jd_embd = self.embd(jd_tensor)
        return cat_token_embd,exp_embd[1].unsqueeze(1),jd_embd[1].unsqueeze(1)

class Co_FC(nn.Module):
    '''
    对continuous features的Embedding输入到FC
    '''
    def __init__(self,in_size, hiden_size, out_size):
        super(Co_FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size,hiden_size[0]),
            nn.ReLU(),
            nn.Linear(hiden_size[0],hiden_size[1]),
            nn.ReLU(),
            nn.Linear(hiden_size[1],out_size)
        )
    def forward(self,x):
        x = self.fc(x)
        return x

class MatchNet(nn.Module):
    def __init__(self,config,model_config):
        super(MatchNet, self).__init__()
        self.config = config
        # embedding
        self.co_embd_net = CoEmbdNet(config)
        self.ca_embd_net = CaEmbdNet(config)
        self.albert_emd_net = AlbertEmbd(config,model_config)
        # local match each return a tensor(embdsize)
        self.match_years = LocalMatch(4 * config.embdsize, config.embdsize, dropout=config.local_dropout)
        self.match_degree = LocalMatch(4 * config.embdsize, config.embdsize, dropout=config.local_dropout)
        self.match_salary = LocalMatch(4 * config.embdsize, config.embdsize, dropout=config.local_dropout)
        self.match_city = LocalMatch(4 * config.embdsize, config.embdsize, dropout=config.local_dropout)
        self.match_industry = LocalMatch(4 * config.albert_size, config.albert_size, dropout=config.local_dropout)
        self.match_type = LocalMatch(4 * config.albert_size, config.albert_size, dropout=config.local_dropout)
        self.match_text = LocalMatch(4 * config.albert_size, config.albert_size, dropout=config.local_dropout)
        # preject the text embedding into low dimension
        self.fc = nn.Sequential(
            nn.Linear(config.albert_size, config.hiden_size[1]),
            nn.ReLU(),
            nn.Linear(config.hiden_size[1], config.hiden_size[2]),
            nn.ReLU(),
            nn.Linear(config.hiden_size[2], config.embdsize),
            nn.ReLU()
        )
        # MultiheadAtt
        self.matt = MultiheadAtt(config.embdsize,config.num_heads)
        # mlp
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(7*config.embdsize, 2*config.embdsize),
            nn.ReLU(),
            nn.Linear(2*config.embdsize, config.embdsize),
            nn.ReLU(),
            nn.Linear(config.embdsize, 1),
        )

    def forward(self,co_xi, co_xv, city_xi, city_xv, cat_token_tensors, exp_tensor, jd_tensor):
        # get the embedding vecotr of each feature field
        co_embd = self.co_embd_net(co_xi, co_xv) # - *11*config.embdsize
        ca_embd = self.ca_embd_net(city_xi, city_xv) # - *5*config.embdsize
        # -*9*312,-*1*312,-*1*312
        ca_token_embd, exp_embd, jd_embd = self.albert_emd_net(cat_token_tensors,exp_tensor,jd_tensor)
        # years local match
        years_a = co_embd[:, [3], :]
        years_b = co_embd[:, [4], :]
        years = self.match_years(years_a,years_b) # - * config.embdsize
        # edu local match
        edu_a = co_embd[:, [5], :]
        edu_b = co_embd[:, [6], :]
        edu = self.match_degree(edu_a,edu_b)
        # salary local match, the num 9 denote the job's max salary
        # he num 10 denote the job's max salary
        salary_a = co_embd[:, [7,8], :]
        salary_b = co_embd[:, [9], :]
        # salary_b = co_embd[:, [10], :]
        l = salary_a.size()[1]
        salary_b = salary_b.repeat(1,l,1)
        salary = self.match_degree(salary_a, salary_b)
        # city local match
        city_a = ca_embd[:, [0,1,2,3], :]
        city_b = ca_embd[:, [4], :]
        l = city_a.size()[1]
        city_b = city_b.repeat(1, l, 1)
        city = self.match_city(city_a,city_b)
        # industry local match
        industry_a = ca_token_embd[:, [0, 1, 2, 3], :]
        industry_b = ca_token_embd[:, [4], :]
        l = industry_a.size()[1]
        industry_b = industry_b.repeat(1, l, 1)
        industry = self.match_industry(industry_a, industry_b)
        # type local match
        type_a = ca_token_embd[:, [5, 6, 7], :]
        type_b = ca_token_embd[:, [8], :]
        l = type_a.size()[1]
        type_b = type_b.repeat(1, l, 1)
        type = self.match_type(type_a, type_b)
        # text local match
        text = self.match_text(exp_embd,jd_embd)
        # preject the text vectors into low dimension
        text_vec = torch.cat([industry, type, text],1).view(-1,3,self.config.albert_size)
        text_vec = self.fc(text_vec)
        # concatenate the local match vectors
        cat_vec = torch.cat([years, edu, salary, city, ], 1).view(-1,4,self.config.embdsize)
        feat_vec = torch.cat([cat_vec,text_vec],1) # - *7*config.embdsize
        # multi-head self fattention
        matt_vec, matt_weights = self.matt(feat_vec)
        # residual layer
        in_vec = (feat_vec + matt_vec).view(-1,7*self.config.embdsize)
        # input into the mlp
        pre = self.mlp(in_vec)
        return pre,matt_weights

class MultiheadAtt(nn.Module):
    '''
    借助nn.MultiheadAttention实现多头注意力机制,其实现过程中有相应的weights
    '''
    def __init__(self,ebd_size,num_heads):
        super(MultiheadAtt,self).__init__()
        self.matt = nn.MultiheadAttention(ebd_size,num_heads)
    def forward(self,x):
        x = x.transpose(0, 1)
        matt_out, matt_weights = self.matt(x, x, x)
        return matt_out.transpose(0,1),matt_weights
        
class LocalMatch(nn.Module):
    def __init__(self,insize,outsize,dropout):
        super(LocalMatch, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(insize, insize),
            # nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(insize,outsize),
            nn.PReLU(),
        )
    def forward(self,a,b, pool='max'):
        # 参数pool设置不同效果可能不同
        c = torch.cat([a, b, a - b, a * b], dim=-1)
        # c = torch.cat([a, b], dim=-1)
        c = self.net(c)
        # c = c.unsqueeze(-1)
        if pool.lower() == 'max':
            c = c.max(dim=1).values
        elif pool.lower() == 'mean':
            c = c.mean(dim=1)
        elif pool.lower() == 'sum':
            c = c.sum(dim=1)
        return c

    
