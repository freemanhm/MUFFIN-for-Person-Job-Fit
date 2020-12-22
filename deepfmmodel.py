# deepFM model
import torch
import torch.nn as nn
# from config import DefaultConfig
# config = DefaultConfig()

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


class DeepFM(nn.Module):
    '''
    deepFM model
    '''

    def __init__(self,config):
        super(DeepFM, self).__init__()
        self.config = config
        self.ca_feat_size = self.get_cat_size()# 原来在config中写的ca_feat_size
        # ca_feat_size = [len(feature2idx['city']),len(feature2idx['industry']), len(feature2idx['type']),len(feature2idx['travel'])]
        self.embedding_size = config.embedding_size
        self.co_num = config.fmco_num
        self.ca_num = config.fmca_num
        self.field_num = torch.tensor(
            [self.co_num, self.co_num + 5, self.co_num + 10, self.co_num + 15, self.co_num + 16])
        # 一阶部分 定义权重矩阵
        # for the continuous features
        self.first_co_emb = nn.Embedding(self.co_num + 1, 1)
        nn.init.xavier_uniform_(self.first_co_emb.weight)
        # for category features
        self.first_ca_emb = nn.ModuleList(
            [nn.Embedding(feat_size, self.embedding_size) for feat_size in self.ca_feat_size])
        for i in range(len(self.first_ca_emb)):
            nn.init.xavier_uniform_(self.first_ca_emb[i].weight)

            # 二阶部分 FM
        self.second_co_emb = nn.Embedding(self.co_num + 1, self.embedding_size)
        nn.init.xavier_uniform_(self.second_co_emb.weight)
        self.second_ca_emb = nn.ModuleList(
            [nn.Embedding(feat_size, self.embedding_size) for feat_size in self.ca_feat_size])
        for i in range(len(self.second_ca_emb)):
            nn.init.xavier_uniform_(self.second_ca_emb[i].weight)

        # deep 部分
        # 共享了第二部分的Embedding层
        self.deep = Deep(
            input_size=(self.co_num + self.ca_num) * self.embedding_size,
            output_size=self.embedding_size
        )

        # 最后一层 fc
        self.fc = nn.Linear(self.embedding_size * 2 + 1, 1)

    def get_cat_size(self):
        config = self.config
        with open(config.feature2idx_path,'r') as file:
            dicts = eval(file.read())
        return [len(dicts['city']),len(dicts['industry']), len(dicts['type']),len(dicts['travel'])]

    def forward(self, xi, xv):
        # 一阶部分
        # for continuous features
        first_co_emb = self.first_co_emb(xi[:, 0:self.co_num])
        first_co_value = torch.mul(first_co_emb, xv[:, 0:self.co_num].unsqueeze(-1))
        # for category features
        first_ca_value = [0 for feat_size in self.ca_feat_size]

        for i in range(len(self.first_ca_emb)):
            first_idx = self.first_ca_emb[i](xi[:, self.field_num[i]:self.field_num[i + 1]])
            first_ca_value[i] = torch.mul(first_idx, xv[:, self.field_num[i]:self.field_num[i + 1]].unsqueeze(-1))
            # concat category features
        first_ca_value = torch.cat([x for x in first_ca_value], 1)
        # sum the category features
        first_ca_value = torch.sum(first_ca_value, dim=2).unsqueeze(-1)
        # concat continuous features and category features
        first_value = torch.cat((first_co_value, first_ca_value), 1)
        # sum the all features
        y_first_order = torch.sum(first_co_value, dim=1)  # batch_size*1
        #         print(y_first_order.size())
        # 二阶部分 FM
        # for continuous feature
        second_co_idx = self.second_co_emb(xi[:, 0:self.co_num])
        second_co_value = torch.mul(second_co_idx, xv[:, 0:self.co_num].unsqueeze(-1))
        # for category features
        second_ca_value = [0 for feat_size in self.ca_feat_size]
        for i in range(len(self.first_ca_emb)):
            second_idx = self.second_ca_emb[i](xi[:, self.field_num[i]:self.field_num[i + 1]])
            second_ca_value[i] = torch.mul(second_idx, xv[:, self.field_num[i]:self.field_num[i + 1]].unsqueeze(-1))
            # concat the category
        second_ca_value = torch.cat([x for x in second_ca_value], 1)
        second_value = torch.cat((second_co_value, second_ca_value), 1)  # batch_size*ca_num*emb_size
        # sum_square part
        summed_feat_emb = torch.sum(second_value, dim=1)  # batch_size*ca_num*emb_size
        interaction_part1 = torch.pow(summed_feat_emb, 2)  # batch_size*ca_num*emb_size
        # squared_sum part
        squared_feat_emd_value = torch.pow(second_value, 2)
        interaction_part2 = torch.sum(squared_feat_emd_value, dim=1)
        y_secd_order = 0.5 * torch.sub(interaction_part1, interaction_part2)

        # deep 部分
        y_deep = second_value.reshape(-1, xi.size()[1] * self.embedding_size)  # batch_size * (num_feat* emb_size)
        y_deep = self.deep(y_deep).squeeze(1)

        # last layer
        # concat the three parts
        concat_input = torch.cat((y_first_order, y_secd_order, y_deep), dim=1)
        y_pre = self.fc(concat_input)

        return y_pre



