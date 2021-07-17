import torch

class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_path = "embedding"
        self.train_path = "data/train.json"
        self.dev_path = "data/dev.json"

        self.batch_size = 64
        self.epoch_size = 10
        self.pad_size = 256
        self.learning_rate = 5e-5
        self.weight_decay = 1e-2