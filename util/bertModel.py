import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        
        self.bert = BertModel.from_pretrained(config.bert_path)
#         for param in self.bert.parameters():
#             param.requires_grad = False
        
        # self.fc = nn.Linear(768, 384)
        self.output = nn.Linear(768, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.bert(x, output_all_encoded_layers=False)[1]
        
#         out = self.fc(out)
#         out = torch.relu(out)
        out = self.output(out)
        out = self.dropout(out)
        
        out = torch.sigmoid(out)

        return out