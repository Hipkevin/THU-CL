from sklearn.metrics import f1_score, classification_report


if __name__ == '__main__':

    with open('data/train.json', 'r', encoding='utf-8') as file:
        text = file.read()

    data = eval(text)

    # 计算macro-F1
    Y = list()
    Y_hat = list()

    # 计算样本个数
    count = 0

    # 保存判错样本
    FP_hold = list()
    FN_hold = list()

    for sample in data:
        defendantSet = sample['被告人集合']
        defendants = sample['被告人']
        sentence = sample['句子']

        count += len(defendantSet)

        for defendant in defendantSet:
            # 真实值
            if defendant in defendants:
                Y.append(1)
            else:
                Y.append(0)

            # 估计值
            if defendant in sentence:
                Y_hat.append(1)
            else:
                Y_hat.append(0)

            # 混淆矩阵
            if Y[-1] != Y_hat[-1]:
                hold = [sentence, defendant]
                if Y[-1] == 1:
                    FN_hold.append(hold)
                else:
                    FP_hold.append(hold)

    print(classification_report(Y, Y_hat))

    print("样本量&macro-F1")
    print(count)
    print(sum(Y))
    print(f1_score(Y, Y_hat, average='macro'))

    print("假正类：")
    print(len(FP_hold))
    print(FP_hold[0])

    print("假负类：")
    print(len(FN_hold))
    print(FN_hold[0])