import torch

from torch.utils.data import DataLoader
from config import Config
from util.bertModel import BERT
from util.dataTool import SeqClsDataSet, getData
from util.trainer import train, evaluate


if __name__ == '__main__':
    config = Config()

    print("Data loading...")
    train_data = SeqClsDataSet(config.train_path, config)
    dev_data = SeqClsDataSet(config.dev_path, config)

    train_loader = DataLoader(train_data, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
    dev_loader = DataLoader(dev_data, batch_size=config.batch_size,
                            shuffle=True, pin_memory=True, num_workers=4, drop_last=False)

    print("Model loading...")
    model = BERT(config).to(config.device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=config.learning_rate, weight_decay=config.weight_decay)

    print("Training...")
    model = train(model,
                  loader=train_loader,
                  criterion=criterion,
                  optimizer=optimizer,
                  config=config)

    print("Testing...", round(len(dev_data)/config.batch_size))
    evaluate(model, dev_loader, config)

    # # get bad case
    # y = list()
    # y_pre = list()
    # for idx, (X, Y) in enumerate(train_loader):
    #     X = X.to(config.device)
    #     Y = Y.to(config.device)
    #
    #     y_pre += model(X).squeeze().cpu().tolist()
    #     y += Y.cpu().tolist()
    #
    # index = list()
    # for i in range(len(y)):
    #     if y_pre[i] > 0.5:
    #         y_hat = 1.0
    #     else:
    #         y_hat = 0.0
    #
    #     if y_hat != y[i][0]:
    #         index.append(i)
    #
    # sample, _ = getData(config.train_path)
    # with open("error_sample.txt", "a+", encoding='utf-8') as file:
    #     for i in index:
    #         file.write(str(sample[i]) + "\n")

    # torch.save(model.state_dict(), 'model/BERT_SeqCls.pt')