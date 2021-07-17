

if __name__ == '__main__':
    with open("error_sample.txt", 'r', encoding='utf-8') as file:
        text = file.read().split('\n')

    for t in text:
        print(len(t))