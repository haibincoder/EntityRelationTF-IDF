import joblib

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


if __name__=="__main__":
    # 定义类别列表
    LABEL_INDEX = ['Cause-Effect', 'Instrument-Agency', 'Product-Producer',
                   'Content-Container', 'Entity-Origin', 'Entity-Destination',
                   'Component-Whole', 'Member-Collection', 'Message-Topic', 'Other']

    model = joblib.load("temp/models/base_model.joblib")
    vectorizer = joblib.load("temp/models/base_vectorizer.joblib")

    text_list = []
    entity_list = []
    with open('temp/test_generate_easy.txt', 'r', encoding='utf-8') as f:
        for line in f:
            data = line.strip().split('\t')
            entity_list.append((data[0], data[1]))
            text_list.append(data[3])
    print('read_finish')

    x = vectorizer.transform(text_list)
    y = model.predict(x)
    with open('temp/results.txt', 'w', encoding='utf-8') as f:
        for line in y:
            f.write(str(line) + '\n')

    for i in range(10):
        print("文本：{}；\n实体一：{}；实体二：{}，关系预测：{}\n" \
              .format(text_list[i], entity_list[i][0], entity_list[i][1], LABEL_INDEX[int(y[i])]))

    print('finish')
