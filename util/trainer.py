from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

def train(model, loader, criterion, optimizer, config):
    model.train()
    for epoch in range(config.epoch_size):

        for idx, (X, Y) in enumerate(loader):
            X = X.to(config.device)
            Y = Y.to(config.device)
            
            predict = model(X)
            loss = criterion(predict, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10:
                print(f"Epoch: {epoch} batch: {idx} | loss: {loss}")

    return model

def evaluate(model, loader, config):
    model.eval()

    y = list()
    y_pre = list()
    for idx, (X, Y) in tqdm(enumerate(loader)):
        X = X.to(config.device)
        Y = Y.to(config.device)
        
        y_pre += model(X).squeeze().cpu().tolist()
        y += Y.cpu().tolist()

    y_hat = list()
    for pre in y_pre:
        if pre > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)

    print(classification_report(y, y_hat))

    macro_f1 = f1_score(y_true=y, y_pred=y_hat, average="macro")
    print("macro-F1: ", macro_f1)