import torch

from torch.utils.data import DataLoader
from config import Config
from util.bertModel import BERT
from util.dataTool import SeqClsDataSet
from util.trainer import train, evaluate


if __name__ == '__main__':
    config = Config()

    print("Data loading...")
    train_data = SeqClsDataSet(config.train_path, config)
    dev_data = SeqClsDataSet(config.dev_path, config)

    train_loader = DataLoader(train_data, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
    dev_loader = DataLoader(dev_data, batch_size=len(dev_data) // 4,
                            shuffle=True, pin_memory=True, num_workers=4, drop_last=False)

    print("Model loading...")
    model = BERT(config)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate, weight_decay=config.weight_decay)

    # print("Training...")
    # model = train(model,
    #               loader=train_loader,
    #               criterion=criterion,
    #               optimizer=optimizer,
    #               config=config)

    print("Testing...")
    evaluate(model, dev_loader)
