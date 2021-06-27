def getData(path):
    """
    从json中读取数据，解析被告人集合、被告人、句子、要素原始值等字段
    并构造样本：（句子*，被告人）-（是否存在关联）

    :param path: 文件路径
    :return: 返回X和Y
    """
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read().split('\n')
        text.pop(-1)

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