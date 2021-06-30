class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.bert_path = "embedding"
        self.train_path = "data/test.json"
        self.dev_path = "data/test.json"

        self.batch_size = 16
        self.epoch_size = 2
        self.pad_size = 256
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4