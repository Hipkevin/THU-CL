import jieba
import re
import torch

from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from pytorch_pretrained_bert import BertTokenizer

# BERT token常量
CLS = "[CLS]"
SEP = "[SEP]"
PAD = "[PAD]"


def getData(path):
    """
    从json中读取数据，解析被告人集合、被告人、句子、要素原始值等字段
    并构造样本：（句子*，被告人）-（是否存在关联）

    :param path: 文件路径
    :return: 返回X和Y
    """
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()
        text = eval(text)

    sample = list()
    label = list()
    for t in text:
        defendantSet = t['被告人集合']
        defendants = t['被告人']
        sentence = t['句子']
        original_factor = t["要素原始值"]

        for defendant in defendantSet:
            sample.append([sentence, defendant, original_factor])

            if defendant in defendants:
                label.append(1)
            else:
                label.append(0)

    return sample, label


def get_new_segment(segment):  # 新增的方法 ####
    """
    输入一句话，返回一句经过处理的话: 为了支持中文全称mask，将被分开的词，将上特殊标记("#")，使得后续处理模块，能够知道哪些字是属于同一个词的。
    :param segment: 一句话. e.g.  ['悬', '灸', '技', '术', '培', '训', '专', '家', '教', '你', '艾', '灸', '降', '血', '糖', '，', '为', '爸', '妈', '收', '好', '了', '！']
    :return: 一句处理过的话 e.g.    ['悬', '##灸', '技', '术', '培', '训', '专', '##家', '教', '你', '艾', '##灸', '降', '##血', '##糖', '，', '为', '爸', '##妈', '收', '##好', '了', '！']
    """
    seq_cws = jieba.lcut("".join(segment))  # 分词
    seq_cws_dict = {x: 1 for x in seq_cws}  # 分词后的词加入到词典dict
    new_segment = []
    i = 0
    while i < len(segment):  # 从句子的第一个字开始处理，知道处理完整个句子
        if len(re.findall('[\u4E00-\u9FA5]', segment[i])) == 0:  # 如果找不到中文的，原文加进去即不用特殊处理。
            new_segment.append(segment[i])
            i += 1
            continue

        has_add = False
        for length in range(3, 0, -1):
            if i + length > len(segment):
                continue
            if ''.join(segment[i:i + length]) in seq_cws_dict:
                new_segment.append(segment[i])
                for l in range(1, length):
                    new_segment.append('##' + segment[i + l])
                i += length
                has_add = True
                break
        if not has_add:
            new_segment.append(segment[i])
            i += 1

    return new_segment


class SeqClsDataSet(Dataset):
    def __init__(self, path, config):
        super(SeqClsDataSet, self).__init__()
        sample, label = getData(path)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)

        X = list()
        Y = list()
        for i in tqdm(range(len(sample))):
            """
            defendants = t['被告人']
            sentence = t['句子']
            original_factor = t["要素原始值"]
            sample.append([sentence, defendant, original_factor])
            """
            data = sample[i]
            l = label[i]

            seq = CLS + data[1] + data[2] + SEP + data[0] + SEP
            seq_token = self.tokenizer.tokenize(seq)

            if len(seq_token) > config.pad_size:
                seq_token = seq_token[:config.pad_size]
            else:
                seq_token += [PAD]*(config.pad_size-len(seq_token))

            seq_id = self.tokenizer.convert_tokens_to_ids(seq_token)

            X.append(seq_id)
            Y.append(torch.tensor(l))

        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)