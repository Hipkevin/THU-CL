import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()

        self.bert = BertModel.from_pretrained(config.bert_path)
        self.output = nn.Linear(768, 1)

    def forward(self, x):
        with torch.no_grad():
            out = self.bert(x)[1]

        return self.output(out)