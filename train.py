import joblib

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

seed = 2020


def read_data(filepath):
    # https://blog.csdn.net/qq_29883591/article/details/88567561
    labels_dict = {'Cause-Effect': '0', 'Instrument-Agency': '1', 'Product-Producer': '2',
                   'Content-Container': '3', 'Entity-Origin': '4', 'Entity-Destination': '5',
                   'Component-Whole': '6', 'Member-Collection': '7', 'Message-Topic': '8',
                   'Other': '9'}

    label_list, text_list = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip().split('\t')
            label_list.append(labels_dict[text[2]])
            text_list.append(text[3])

    return text_list, label_list


if __name__=="__main__":
    x_train, y_train = read_data('temp/train_generate_easy.txt')
    x_test, y_test = read_data('temp/test_generate_easy.txt')

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=seed)
    print("train data: {}, val data: {}, test data: {}".format(len(y_train), len(y_val), len(y_test)))

    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    x = [i for i in range(len(labels))]
    count_in_train = [y_train.count(label) / len(y_train) for label in labels]  # 统计训练集占比
    count_in_val = [y_val.count(label) / len(y_val) for label in labels]
    count_in_test = [y_test.count(label) / len(y_test) for label in labels]
    plt.figure()
    plt.bar(x, count_in_train, width=0.3, label="train_set")
    plt.bar([i + 0.3 for i in x], count_in_val, width=0.3, label="val_set")
    plt.bar([i + 0.6 for i in x], count_in_test, width=0.3, label="test_set")
    plt.xticks([i + 0.3 for i in x], labels)
    plt.legend()
    # plt.show()
    print('show plt')

    # 构建tf-idf词典
    # 默认匹配长度大于1的单词，对于中文需要特殊处理，
    tfidf_model = TfidfVectorizer(stop_words='english').fit(x_train)  # 构造tfidf向量
    print("词典大小 {}".format(len(tfidf_model.vocabulary_)))

    # 构造tf-idf向量
    # 得到tf-idf矩阵，稀疏矩阵表示法
    x_train_vec = tfidf_model.transform(x_train)
    x_val_vec = tfidf_model.transform(x_val)

    x_temp = x_train_vec[:1]
    print(x_temp)
    # 高维稀疏矩阵
    #  (0, 8880)	0.6182306565570105  第0个字符，对应词典序号为8880的词的TFIDF为0.618230...
    #  (0, 8250)	0.3761725985922683
    #  (0, 4698)	0.4674077584472665
    #  (0, 1982)	0.507754880532654
    print(x_temp.todense())
    # 转换为一般矩阵
    # [[0. 0. 0. ... 0. 0. 0.]]

    # print(type(x_train_vec), x_train_vec)

    # 定义模型
    clf = LogisticRegression(solver='lbfgs')
    # 训练模型，warning信息可以忽略
    clf.fit(x_train_vec, y_train)
    print("Train DONE")

    import os

    if not os.path.exists('temp/models'):
        os.makedirs('temp/models')
    joblib.dump(clf, "temp/models/base_model.joblib")
    joblib.dump(tfidf_model, "temp/models/base_vectorizer.joblib")

    # 计算准确率
    y_ = clf.predict(x_val_vec)
    acc = accuracy_score(y_val, y_)
    print("验证集准确率: {}".format(acc))
