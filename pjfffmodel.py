# deepFM model
import torch
import torch.nn as nn


class Deep(nn.Module):
    def __init__(self, input_size, output_size):
        super(Deep, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class UserFM(nn.Module):
    '''
    user FM side hidden vector
    '''
    def __init__(self,config):
        super(UserFM,self).__init__()
        self.config = config
        self.ca_feat_size = self.get_cat_size()
        # 一阶linear部分, 加权求和
        self.firt_linear = nn.Linear(16,1)
        # 二阶部分的Embedding
        self.ca_emb = nn.ModuleList(
            [nn.Embedding(feat_size, self.config.embedding_size) for feat_size in self.ca_feat_size])
        for i in range(len(self.ca_emb)):
            nn.init.xavier_uniform_(self.ca_emb[i].weight)
        self.deep = Deep(3*self.config.embedding_size, self.config.embedding_size)# 输入输出维度待定
        self.fc = nn.Linear(2*self.config.embedding_size+1,self.config.embedding_size)

    def forward(self,co_xi,co_xv,ca_xi,ca_xv):
        userco = [1, 3, 5, 7, 8]
        userca = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12]
        # 拼接一阶线性部分
        first_x = torch.cat((co_xv[:,userco].float(),ca_xv[:,userca].float()),dim=1)
        first_v = self.firt_linear(first_x)
        # 二阶 FM 部分
        ca_value = [0 for feat_size in self.ca_feat_size]
        # for user city
        city_embds = self.ca_emb[0](ca_xi[:, 0:4])
        ctiy_embd = city_embds.mean(dim=1)
        # for industry
        industry_embds = self.ca_emb[1](ca_xi[:,5:9])
        industry_embd = industry_embds.mean(dim=1)
        # for type
        type_embds = self.ca_emb[2](ca_xi[:,10:13])
        typ_embd = type_embds.mean(dim=1)
        # 二阶FM值计算
        second_embd = torch.cat([ctiy_embd,industry_embd,typ_embd],dim=1).view(-1,3,self.config.embedding_size)
        # sum_square part
        summed_feat_emb = torch.sum(second_embd, dim=1)  # batch_size*ca_num*emb_size
        interaction_part1 = torch.pow(summed_feat_emb, 2)  # batch_size*ca_num*emb_size
        # squared_sum part
        squared_feat_emd_value = torch.pow(second_embd, 2)
        interaction_part2 = torch.sum(squared_feat_emd_value, dim=1)
        second_v = 0.5 * torch.sub(interaction_part1, interaction_part2)

        # deep 部分
        y_deep = second_embd.reshape(-1, 3 * self.config.embedding_size)  # batch_size * (num_feat* emb_size)
        deep_v = self.deep(y_deep).squeeze(1)

        # last layer
        # concat the three parts
        concat_input = torch.cat((first_v, second_v, deep_v), dim=1)
        fm_user = self.fc(concat_input)
        return fm_user

    def get_cat_size(self):
        config = self.config
        with open(config.feature2idx_path, 'r') as file:
            dicts = eval(file.read())
        return [len(dicts['city']), len(dicts['industry']), len(dicts['type'])]

class JobFM(nn.Module):
    '''
    user FM side hidden vector
    '''
    def __init__(self,config):
        super(JobFM,self).__init__()
        self.config = config
        self.ca_feat_size = self.get_cat_size()
        # 一阶linear部分, 加权求和
        self.firt_linear = nn.Linear(8,1)
        # 二阶部分的Embedding
        self.ca_emb = nn.ModuleList(
            [nn.Embedding(feat_size, self.config.embedding_size) for feat_size in self.ca_feat_size])
        for i in range(len(self.ca_emb)):
            nn.init.xavier_uniform_(self.ca_emb[i].weight)
        self.deep = Deep(2*self.config.embedding_size, self.config.embedding_size)# 输入输出维度待定
        self.fc = nn.Linear(2*self.config.embedding_size+1,self.config.embedding_size)

    def forward(self,co_xi,co_xv,ca_xi,ca_xv):
        userco = [1, 3, 5, 7, 8]
        userca = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12]
        jobco = [0, 2, 4, 6, 9, 10]
        jobca = [4, 13]# job 没有industry，现在是title
        # 拼接一阶线性部分
        first_x = torch.cat((co_xv[:,jobco].float(),ca_xv[:,jobca].float()),dim=1)
        first_v = self.firt_linear(first_x)
        # 二阶 FM 部分
        ca_value = [0 for feat_size in self.ca_feat_size]
        # for user city
        city_embd = self.ca_emb[0](ca_xi[:, [4]])
        # ctiy_embd = city_embds.mean(dim=1)
        # for industry
        # industry_embds = self.ca_emb[1](ca_xi[:,5:9])
        # industry_embd = industry_embds.mean(dim=1)
        # for type
        type_embd = self.ca_emb[1](ca_xi[:,[13]])
        # typ_embd = type_embds.mean(dim=1)
        # 二阶FM值计算
        second_embd = torch.cat([city_embd,type_embd],dim=1).view(-1,2,self.config.embedding_size)
        # sum_square part
        summed_feat_emb = torch.sum(second_embd, dim=1)  # batch_size*ca_num*emb_size
        interaction_part1 = torch.pow(summed_feat_emb, 2)  # batch_size*ca_num*emb_size
        # squared_sum part
        squared_feat_emd_value = torch.pow(second_embd, 2)
        interaction_part2 = torch.sum(squared_feat_emd_value, dim=1)
        second_v = 0.5 * torch.sub(interaction_part1, interaction_part2)

        # deep 部分
        y_deep = second_embd.reshape(-1, 2 * self.config.embedding_size)  # batch_size * (num_feat* emb_size)
        deep_v = self.deep(y_deep).squeeze(1)

        # last layer
        # concat the three parts
        concat_input = torch.cat((first_v, second_v, deep_v), dim=1)
        fm_job = self.fc(concat_input)
        return fm_job

    def get_cat_size(self):
        config = self.config
        with open(config.feature2idx_path, 'r') as file:
            dicts = eval(file.read())
        return [len(dicts['city']), len(dicts['type'])]

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

class SentVec(nn.Module):
    def __init__(self,config):
        super(SentVec, self).__init__()
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
        return resume_vec, job_vec

class MLP(nn.Module):
    def __init__(self, insize,config):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(insize, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Classfier(nn.Module):
    def __init__(self, insize,config):
        super(Classfier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(insize, config.hiden_size[1]),
            nn.ReLU(),
            nn.Linear(config.hiden_size[1], config.hiden_size[2]),
            nn.ReLU(),
            nn.Linear(config.hiden_size[2],1)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class PJFFF(nn.Module):
    def __init__(self,config):
        super(PJFFF,self).__init__()
        self.config = config
        self.userfm = UserFM(config)
        self.jobfm = JobFM(config)
        self.sentvec = SentVec(config)
        insize = config.dim+config.embedding_size
        # self.classfier = Classfier(insize,insize,config)
        self.u_mlp = MLP(insize, config)
        self.j_mlp = MLP(insize, config)
        self.classfier = Classfier(256,config)
    def forward(self,co_xi, co_xv, cat_xi, cat_xv, exp_sent, jd_sent):
        fm_u = self.userfm(co_xi, co_xv, cat_xi, cat_xv)
        fm_j = self.jobfm(co_xi, co_xv, cat_xi, cat_xv)
        sentvec_u, sentvec_j = self.sentvec(exp_sent, jd_sent)
        u_in = torch.cat((fm_u,sentvec_u),dim=1)
        j_in = torch.cat((fm_j,sentvec_j),dim=1)
        # user 和job 分别输入一个mlp得到一个128维度向量表示
        u_vec = self.u_mlp(u_in)
        j_vec = self.j_mlp(j_in)
        xin = torch.cat((u_vec,j_vec),dim=1)
        s = self.classfier(xin)
        return s

