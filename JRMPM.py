import re
import jieba
import torch
import numpy as np
import gzip
from torch.autograd import Variable
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as f
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import os
import torch.nn.utils.rnn  as rnn
import torch.nn.utils.rnn as rnn_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
max_pos_num = 2
max_neg_num = 3
max_his_num = 5
max_sen_len = 50 # 每句话50个词
doc_max_len = 20 # 每个doc20句话
vocab_size = 90000
emb_dim = 100
hidden_dim = 100
dropout_rate = 0.2
epoch_num = 100
batch_size = 128
step_per_epoch = 800
valid_size = 1280
dir = './data/jrmpm_data'

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.sen_maxl = max_sen_len
        self.his_maxl = doc_max_len
        self.total_word = vocab_size + 2
        self.embedding_dim = emb_dim
        self.embedding = nn.Embedding(self.total_word, self.embedding_dim, padding_idx=0)

    def forward(self, ids):
        ids = Variable(torch.LongTensor(ids), requires_grad=False).cuda().view(-1, self.sen_maxl)
        emb = self.embedding(ids)
        length = self.get_length(ids)
        return emb, length

    def get_length(self, ids):
        length = (1 + torch.abs(2 * torch.sum((ids != 0), 1).long() - 1)) / 2
        return length.cpu()

class Sen_Rnn(nn.Module):
    def __init__(self):
        super(Sen_Rnn, self).__init__()
        self.sen_maxl = max_sen_len
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, emb, length):
        length_order, index_order = torch.sort(length, descending=True)
        emb_order = emb[index_order]
        reverse_order = torch.sort(index_order)[1]
        pack_emb = rnn_utils.pack_padded_sequence(emb_order, length_order, batch_first=True)
        out, hidden = self.rnn(pack_emb)
        hidden = hidden.squeeze(0)
        return hidden[reverse_order]


class Doc_Rnn(nn.Module):
    def __init__(self):
        super(Doc_Rnn, self).__init__()
        self.doc_maxl = doc_max_len
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=50, bidirectional=True)

    def forward(self, sen_rep, length):
        '''

        :param sen_rep: -1 * doc_len * hidden
        :return:
        '''
        length = torch.LongTensor(length).view(-1)
        length_order, index_order = torch.sort(length, descending=True)
        emb_order = sen_rep[index_order]
        reverse_order = torch.sort(index_order)[1]
        pack_emb = rnn_utils.pack_padded_sequence(emb_order, length_order, batch_first=True)
        out, hidden = self.rnn(pack_emb)
        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True, total_length=self.doc_maxl)
        return out[reverse_order]


class Reading_memory(nn.Module):
    def __init__(self):
        super(Reading_memory, self).__init__()
        self.hidden_dim = hidden_dim
        self.doc_maxl = doc_max_len
        self.eps = -1e20
        self.transform_mem = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
                                           nn.Tanh())
        self.transform_sub = nn.Sequential(nn.Linear(in_features=2 * self.hidden_dim, out_features=self.hidden_dim),
                                           nn.Tanh())
        self.update_gate = nn.Sequential(nn.Linear(in_features=3 * self.hidden_dim, out_features=self.hidden_dim),
                                         nn.Sigmoid())

    def forward(self, sub_emb, memory, sub_len, sub_emb_raw):
        mem_4att = self.transform_mem(memory)
        sub_4att = self.transform_sub(torch.cat((sub_emb, sub_emb_raw), 2))
        sub_mem_att = torch.bmm(sub_4att, mem_4att.permute(0, 2, 1))
        mem_inf_mask, mem_zero_mask = self.get_mask(sub_len)
        sub_mem_att = mem_zero_mask * f.softmax((mem_inf_mask + sub_mem_att), 2)
        sub_mem = torch.bmm(sub_mem_att, memory)
        update_gate = self.update_gate(torch.cat((sub_emb, sub_mem, sub_mem * sub_emb), 2))
        sub_emb = (1 - update_gate) * sub_emb + update_gate * sub_mem
        return sub_emb

    def get_mask(self, length):
        bsz = len(length)

        inf_mask = torch.zeros(bsz, self.doc_maxl, self.doc_maxl).cuda()
        zero_mask = torch.ones(bsz, self.doc_maxl, self.doc_maxl).cuda()
        for i in range(bsz):
            if length[i] != self.doc_maxl:
                inf_mask[i, length[i]:, :] = self.eps
                zero_mask[i, length[i]:, :] = 0.0

        return inf_mask, zero_mask


class Updating_memory(nn.Module):
    def __init__(self):
        super(Updating_memory, self).__init__()
        self.hidden_dim = hidden_dim
        self.doc_maxl = doc_max_len
        self.eps = -1e20
        self.transform_mem = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
                                           nn.Tanh())
        self.transform_his = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
                                           nn.Tanh())
        self.transform_sub = nn.Sequential(nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
                                           nn.Tanh())
        self.update_gate = nn.Sequential(nn.Linear(in_features=3 * self.hidden_dim, out_features=self.hidden_dim),
                                         nn.Sigmoid())

    def forward(self, his_emb, sub_emb, memory, sub_len, his_len):
        his_4att = self.transform_his(his_emb)
        sub_4att = self.transform_sub(sub_emb)
        memory_4att = self.transform_mem(memory)
        mem_his_att = torch.bmm(memory_4att, his_4att.permute(0, 2, 1))
        mem_sub_att = torch.bmm(memory_4att, sub_4att.permute(0, 2, 1))
        his_inf_mask, his_zero_mask = self.get_mask(his_len)
        sub_inf_mask, sub_zero_mask = self.get_mask(sub_len)
        mem_his_att = his_zero_mask * f.softmax((mem_his_att + his_inf_mask), 2)
        mem_his = torch.bmm(mem_his_att, his_emb)
        mem_sub_att = sub_zero_mask * f.softmax((mem_sub_att + sub_inf_mask), 2)
        mem_sub = torch.bmm(mem_sub_att, sub_emb)
        mem_att = mem_his + mem_sub
        update_gate = self.update_gate(torch.cat((mem_att, memory, mem_att * memory), 2))
        memory = (1 - update_gate) * memory + update_gate * mem_att
        return memory

    def get_mask(self, length):
        bsz = len(length)
        inf_mask = torch.zeros(bsz, self.doc_maxl, self.doc_maxl).cuda()
        zero_mask = torch.ones(bsz, self.doc_maxl, self.doc_maxl).cuda()
        for i in range(bsz):
            if length[i] != self.doc_maxl:
                inf_mask[i, length[i]:, :] = self.eps
                zero_mask[i, length[i]:, :] = 0.0
        return inf_mask, zero_mask


class Classfier(nn.Module):
    def __init__(self):
        super(Classfier, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.Bil = nn.Bilinear(in1_features=self.hidden_dim, in2_features=self.hidden_dim,
                               out_features=self.hidden_dim)
        self.Dropout_bi = nn.Dropout(dropout_rate)
        self.MLP = nn.Sequential(nn.Tanh(),
                                 nn.Linear(in_features=hidden_dim, out_features=1),
                                 nn.Sigmoid())
        self.Dropout_mlp = nn.Dropout(dropout_rate)

    def forward(self, sub, obj):
        feature = self.Bil(sub, obj)
        # feature = self.Dropout_bi(feature)
        match_score = self.MLP(feature).squeeze(1)
        return match_score


class Trainer():
    def __init__(self,dt):
        self.embedding_dim = emb_dim
        self.doc_maxl = doc_max_len
        self.sen_maxl = max_sen_len
        self.valid_size = valid_size
        self.hidden_dim = hidden_dim
        self.his_len = max_his_num
        self.eps = 1e-30

        self.pos_set = dt['pos_set']
        self.neg_set = dt['neg_set']
        self.resume_id = dt['resume_id']
        self.job_id = dt['job_id']
        self.geek_dt = dt['geek_dt']
        self.job_dt = dt['job_dt']
        self.word_emb_dt = dt['wordemb']

        self.Embedding = Embedding().cuda()
        self.Embedding.embedding.weight.data.copy_(self.word_emb_dt)
        self.Job_sen_rnn = Sen_Rnn().cuda()
        self.Geek_sen_rnn = Sen_Rnn().cuda()
        self.Job_doc_rnn = Doc_Rnn().cuda()
        self.Geek_doc_rnn = Doc_Rnn().cuda()
        self.Job_r_memory = Reading_memory().cuda()
        self.Geek_r_memory = Reading_memory().cuda()
        self.Job_u_memory = Updating_memory().cuda()
        self.Geek_u_memory = Updating_memory().cuda()
        self.Classfier = Classfier().cuda()
        self.pos_train, self.pos_valid = self.divide_dataset(self.pos_set)
        self.neg_train, self.neg_valid = self.divide_dataset(self.neg_set)
        self.optimizer = torch.optim.Adam(
            list(self.Job_doc_rnn.parameters())
            + list(self.Job_sen_rnn.parameters())
            + list(self.Geek_doc_rnn.parameters())
            + list(self.Geek_sen_rnn.parameters())
            + list(self.Job_r_memory.parameters())
            + list(self.Job_u_memory.parameters())
            + list(self.Geek_r_memory.parameters())
            + list(self.Geek_u_memory.parameters())
            + list(self.Classfier.parameters())
            , lr=5e-4)

    def train(self):
        train_pos_pair = self.get_batch(self.pos_train)
        train_neg_pair = self.get_batch(self.neg_train)
        score_pos = self.model(train_pos_pair).squeeze()
        score_neg = self.model(train_neg_pair).squeeze()

        # loss = -torch.mean(torch.log(f.sigmoid(score_pos - score_neg)))
        loss = -torch.mean(torch.log(score_pos + self.eps)) - torch.mean(torch.log(1 - score_neg + self.eps))
        # loss = -torch.mean(torch.log(score_pos + self.eps )) - torch.mean(1 - torch.log(score_neg) + self.eps)
        # loss = -torch.mean(torch.log(f.sigmoid(score_pos-score_neg)))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def valid(self):
        self.valid_step = int(self.valid_size / batch_size)
        self.Job_doc_rnn.eval()
        self.Job_sen_rnn.eval()
        self.Geek_doc_rnn.eval()
        self.Geek_sen_rnn.eval()
        self.Embedding.eval()
        # self.Job_cnn.eval()
        self.Job_r_memory.eval()
        self.Job_u_memory.eval()
        # self.Geek_cnn.eval()
        self.Geek_u_memory.eval()
        self.Geek_r_memory.eval()
        self.Classfier.eval()

        match_pos_score = np.zeros(1, dtype=np.float)
        match_nega_score = np.zeros(1, dtype=np.float)
        pos_label = np.ones(self.valid_size, dtype=np.int16)
        neg_label = np.zeros(self.valid_size, dtype=np.int16)
        valid_label = np.hstack((pos_label, neg_label))

        acc = 0.0
        loss = 0.0
        for i in range(self.valid_step):
            valid_pos_pair = self.get_batch(self.pos_valid,
                                            order=i, rand=False)
            valid_neg_pair = self.get_batch(self.neg_valid,
                                            order=i, rand=False)

            score_pos = self.model(valid_pos_pair).squeeze().data
            score_neg = self.model(valid_neg_pair).squeeze().data

            match_pos_score = np.hstack((match_pos_score, score_pos.cpu().data.numpy()))
            match_nega_score = np.hstack((match_nega_score, score_neg.cpu().data.numpy()))
            # loss_batch = -torch.mean(torch.log(f.sigmoid(score_pos-score_neg)))

            # loss_batch = -torch.mean(torch.log(f.sigmoid(score_pos - score_neg)))
            loss_batch = -torch.mean(torch.log(score_pos + self.eps)) - torch.mean(torch.log(1 - score_neg + self.eps))

            # loss_batch = -torch.log(torch.mean(score_pos)) - torch.log(1 - torch.mean(score_neg))
            # loss = torch.mean(score_neg) - torch.mean(score_pos)
            # acc += torch.sum(score_pos - score_neg > 0).data.cpu().numpy()
            loss += loss_batch.data.cpu().numpy()
        match_score = np.hstack((match_pos_score[1:], match_nega_score[1:]))
        match_auc = roc_auc_score(valid_label, match_score)
        match_acc = (np.sum(match_pos_score[1:] > 0.5) + np.sum(match_nega_score[1:] <= 0.5)) / (2 * self.valid_size)
        match_precision = (np.sum(match_pos_score[1:] > 0.5)) / (np.sum(match_score[1:] > 0.5))
        match_recall = (np.sum(match_pos_score[1:] > 0.5)) / self.valid_size
        match_f1 = 2 * match_precision * match_recall / (match_precision + match_recall)
        # print( 'match_auc', match_auc,'loss_match', loss_match)
        acc = acc / self.valid_size
        loss = loss / self.valid_step
        print('valid loss', loss)
        # print('valid acc', acc)
        print('match_auc', match_auc, 'match_acc', match_acc, 'match_pre', match_precision, 'match_rec', match_recall,
              'match_f1', match_f1)

        # print( 'match_auc', match_auc,'loss_match', loss_match)
        acc = acc / self.valid_size
        loss = loss / self.valid_step
        # print('valid loss', loss)
        # print('valid acc', acc)
        # print('match_auc', match_auc)
        self.Embedding.train()
        # self.Job_cnn.train()
        self.Job_r_memory.train()
        self.Job_u_memory.train()
        # self.Geek_cnn.train()
        self.Geek_u_memory.train()
        self.Geek_r_memory.train()
        self.Classfier.train()
        self.Job_doc_rnn.train()
        self.Job_sen_rnn.train()
        self.Geek_doc_rnn.train()
        self.Geek_sen_rnn.train()

    def parse_pair(self, batch_pair):
        geek_content_id = []
        geek_content_length = []
        geek_history_id = []
        geek_history_length = []
        geek_history_num = []
        job_content_id = []
        job_content_length = []
        job_history_id = []
        job_history_length = []
        job_history_num = []
        for pair in batch_pair:
            # [geek_id, job_id] = pair
            [job_id, geek_id] = pair
            geek_dt = self.geek_dt[geek_id]
            geek_content_itemid = self.resume_id[geek_id]['id']
            geek_content_itemlength = self.resume_id[geek_id]['length']
            geek_history = geek_dt['history']

            geek_history_numitem = geek_dt['his_number']
            geek_history_itemids = []
            geek_history_itemlengths = []
            for id in geek_history:
                geek_history_itemid = self.job_id[id]['id']
                geek_history_itemlength = self.job_id[id]['length']
                geek_history_itemids.append(geek_history_itemid)
                geek_history_itemlengths.append(geek_history_itemlength)
            geek_content_id.append(geek_content_itemid)
            geek_content_length.append(geek_content_itemlength)
            geek_history_id.append(geek_history_itemids)
            geek_history_length.append(geek_history_itemlengths)
            geek_history_num.append(geek_history_numitem)

            job_dt = self.job_dt[job_id]
            job_content_itemid = self.job_id[job_id]['id']
            job_content_itemlength = self.job_id[job_id]['length']
            job_history = job_dt['history']
            job_history_numitem = job_dt['his_number']
            job_history_itemids = []
            job_history_itemlengths = []
            job_history_num.append(job_history_numitem)
            for id in job_history:
                job_history_itemid = self.resume_id[id]['id']
                job_history_itemlength = self.resume_id[id]['length']
                job_history_itemids.append(job_history_itemid)
                job_history_itemlengths.append(job_history_itemlength)
            job_content_id.append(job_content_itemid)
            job_content_length.append(job_content_itemlength)
            job_history_id.append(job_history_itemids)
            job_history_length.append(job_history_itemlengths)
        return geek_content_id, geek_content_length, geek_history_id, geek_history_length, geek_history_num, job_content_id, job_content_length, job_history_id, job_history_length, job_history_num

    def representation(self, sub, sub_len, his, his_len, sub_his_num, geek=True):

        # sub_len = torch.LongTensor(sub_len)
        # print(his_len)
        his_len = torch.LongTensor(his_len)
        sub_wordemb, sub_wordemb_len = self.Embedding(sub)  # batchsize * doclen, senlen, embedding
        his_wordemb, his_wordemb_len = self.Embedding(his)  # batchsize * hisnum * doclen * senlen * embedding

        sub_wordemb_3d = sub_wordemb.view(-1, self.sen_maxl, self.embedding_dim)
        his_wordemb_3d = his_wordemb.view(-1, self.sen_maxl, self.embedding_dim)
        # print(sub_wordemb_3d.size(), his_wordemb_3d.size())

        if geek:
            sub_senemb_raw = self.Geek_sen_rnn(sub_wordemb_3d, sub_wordemb_len).view(-1, self.doc_maxl, self.hidden_dim)
            # his_senemb = self.Job_sen_rnn(his_wordemb_3d, his_wordemb_len).view(-1,  self.doc_maxl, self.hidden_dim)
            his_senemb = self.Job_sen_rnn(his_wordemb_3d, his_wordemb_len).view(-1, self.his_len, self.doc_maxl,
                                                                                self.hidden_dim)
            sub_senemb = self.Geek_doc_rnn(sub_senemb_raw, sub_len)
            # his_senemb = self.Job_doc_rnn(his_senemb,his_len).view(-1, self.his_len,self.doc_maxl, self.hidden_dim)
            representation = []
            memory = sub_senemb
            # print(sub_senemb_raw.size(), sub_senemb.size())
            for i in range(self.his_len):
                his_item = his_senemb[:, i, :, :]
                his_item_len = his_len[:, i]
                # print(his_item.size(), his_item_len.size(), sub_senemb.size())
                memory = self.Geek_u_memory(his_item, sub_senemb, memory, sub_len, his_item_len)
                sub_senemb = self.Geek_r_memory(sub_senemb, memory, sub_len, sub_senemb_raw)
                representation.append(sub_senemb)
            representation = torch.stack(representation, 1)  ## batch * his_num * doc_len * hidden_dim
            representation_3d = representation.view(-1, self.doc_maxl, self.hidden_dim)
            representation_useful_index = []

            for i in range(batch_size):
                index = i * self.his_len + (sub_his_num[i] - 1)
                representation_useful_index.append(index)
            representation_useful_index = torch.LongTensor(representation_useful_index).cuda()
            representation_useful = torch.index_select(representation_3d, 0,
                                                       representation_useful_index)  ####  batch * doc * hidden_dim
            representation_useful = torch.max(representation_useful, 1)[0]  ##batch * hidden_dim
            return representation_useful

        if not geek:
            sub_senemb_raw = self.Job_sen_rnn(sub_wordemb_3d, sub_wordemb_len).view(-1, self.doc_maxl, self.hidden_dim)
            # his_senemb = self.Geek_sen_rnn(his_wordemb_3d, his_wordemb_len).view(-1, self.doc_maxl, self.hidden_dim)
            his_senemb = self.Geek_sen_rnn(his_wordemb_3d, his_wordemb_len).view(-1, self.his_len, self.doc_maxl,
                                                                                 self.hidden_dim)
            sub_senemb = self.Job_doc_rnn(sub_senemb_raw, sub_len)
            # his_senemb = self.Geek_doc_rnn(his_senemb, his_len).view(-1, self.his_len, self.doc_maxl, self.hidden_dim)
            representation = []
            memory = sub_senemb
            for i in range(self.his_len):
                his_item = his_senemb[:, i, :, :]
                his_item_len = his_len[:, i]
                memory = self.Job_u_memory(his_item, sub_senemb, memory, sub_len, his_item_len)
                sub_senemb = self.Job_r_memory(sub_senemb, memory, sub_len, sub_senemb_raw)
                representation.append(sub_senemb)
            representation = torch.stack(representation, 1)  ## batch * his_num * doc_len * hidden_dim
            representation_3d = representation.view(-1, self.doc_maxl, self.hidden_dim)
            representation_useful_index = []

            for i in range(batch_size):
                index = i * self.his_len + (sub_his_num[i] - 1)
                representation_useful_index.append(index)
            representation_useful_index = torch.LongTensor(representation_useful_index).cuda()
            representation_useful = torch.index_select(representation_3d, 0,
                                                       representation_useful_index)  ####  batch * doc * hidden_dim
            representation_useful = torch.max(representation_useful, 1)[0]
            return representation_useful

    def model(self, batch_pair):
        geek_content_id, geek_content_length, geek_history_id, geek_history_length, geek_history_num, job_content_id, job_content_length, job_history_id, job_history_length, job_history_num = self.parse_pair(
            batch_pair)
        geek_rep = self.representation(geek_content_id, geek_content_length, geek_history_id, geek_history_length,
                                       geek_history_num)
        job_rep = self.representation(job_content_id, job_content_length, job_history_id, job_history_length,
                                      job_history_num, geek=False)
        score = self.Classfier(geek_rep, job_rep)
        return score

    def divide_dataset(self, dt):
        # return dt, dt
        return dt[self.valid_size:], dt[:self.valid_size]

    def get_batch(self, set, order=0, rand=True):
        length = len(set)
        if rand:
            batch_index = random.sample(range(0, length), batch_size)
        else:
            batch_index = [i for i in range(order * batch_size, (order + 1) * batch_size)]
        batch_pair = []

        for index in batch_index:
            pair = set[index]
            batch_pair.append(pair)
        return batch_pair


if __name__ == '__main__':
    dt = torch.load(dir + '/public.pkl')
    trainer = Trainer()
    for epoch in range(epoch_num):
        for step in range(step_per_epoch):
            total_step = epoch * step_per_epoch + step

            if epoch < epoch_num:
                loss = trainer.train()

            # Logger.scalar_summary('loss', loss.detach().cpu().numpy(), total_step)
            if (step + 1) % 10 == 0:
                # print('sssssss',step)
                # print('epoch',epoch,'step',step,'train_loss',loss.cpu().data.numpy(),'pos_score',pos_score.cpu().data.numpy(),'neg_score',neg_score.cpu().data.numpy())
                print('epoch', epoch, 'step', step, 'loss', loss.cpu().data.numpy())

            if (total_step + 1) % (100) == 0:
                print('valid')
                trainer.valid()
