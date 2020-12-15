# 统一导入工具包
import re  # 正则表达式匹配

# 全局随机种子
seed = 2020


def generate_data(data_dir, name):
    filename = data_dir + name + '/' + name + '.txt'
    e1_list, e2_list, text_list = [], [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            raw_text = line.strip().split('\t')[1]
            entity_1 = re.search(r'<e1>.*</e1>', raw_text).group().replace('<e1>', '').replace('</e1>', '')
            entity_2 = re.search(r'<e2>.*</e2>', raw_text).group().replace('<e2>', '').replace('</e2>', '')
            text = raw_text.replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', '')
            e1_list.append(entity_1)
            e2_list.append(entity_2)
            text_list.append(text)

    label_dir = data_dir + name + '/' + name + '_result.txt'
    label_list = []
    with open(label_dir, 'r', encoding='utf-8') as f:
        for line in f:
            label = line.strip().split('\t')[1]
            label_list.append(label)

    with open('temp/' + name + '_generate_easy.txt', 'w', encoding='utf-8') as f:
        for i in range(len(label_list)):
            f.write(e1_list[i] + '\t' + e2_list[i] + '\t' + label_list[i] + '\t' + text_list[i] + '\n')


# 生成实验数据
generate_data('data/', 'train')
generate_data('data/', 'test')

print('finish')

