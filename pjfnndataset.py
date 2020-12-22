
import sys, os, random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from config import DefaultConfig
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.utils import *

config = DefaultConfig()


# load the data
# token 标识resume 或者job
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


class PJFDataset(Dataset):
    def __init__(self, r_sents, j_sents, token):
        # token用于标注从train，valid，test文件读
        super(PJFDataset, self).__init__()
        self.r_sents = r_sents
        self.j_sents = j_sents
        self.pairs, self.labels = self.load_pairs(token)

    def load_pairs(self, token):
        pairs = []
        labels = []
        assert token in ['train', 'valid', 'test']
        filepath = os.path.join(config.splitdata_path, 'data.{}'.format(token))
        time_print('\nloading from {}'.format(filepath))
        sys.stdout.flush()
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                resume_id, job_id, label = line.strip().split('\t')
                # 删除没有数据（sents）的pairs
                if resume_id not in self.r_sents or job_id not in self.j_sents: continue
                pairs.append([resume_id, job_id])
                labels.append(int(label))
        return pairs, torch.FloatTensor(labels)

    def getlabels(self):
        return self.labels

    def __getitem__(self, index):
        pair = self.pairs[index]
        resume_sent = self.r_sents[pair[0]]
        job_sent = self.j_sents[pair[1]]
        label = self.labels[index]
        return pair, resume_sent, job_sent, label

    def __len__(self):
        return len(self.pairs)


if __name__ == '__main__':
    resume_sents = load_sents('resume')
    job_sents = load_sents("job")
    train_dataset = PJFDataset(resume_sents, job_sents, 'train')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size['train'],
        shuffle=True,
        num_workers=config.num_workers['train']
    )
    for i, (geek_sent, job_sent, labels) in enumerate(train_loader):
        if i < 2:
            print(geek_sent.shape)
            #         print(emb(geek_sent).shape)
            print(job_sent.shape)
            print(len(labels))

