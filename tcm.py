class TCM:
    """
    中医药实体识别数据加载
    """

    def __init__(self):
        # 类别映射

        self.labels = ['SYMPTOM',
                  'DRUG_EFFICACY',
                  'PERSON_GROUP',
                  'SYNDROME',
                  'DRUG_TASTE',
                  'DISEASE',
                  'DRUG_DOSAGE',
                  'DRUG_INGREDIENT',
                  'FOOD_GROUP',
                  'DISEASE_GROUP',
                  'DRUG',
                  'FOOD',
                  'DRUG_GROUP']

        self.id2label = dict(enumerate(self.labels))  # 建立id到类别名的映射
        self.label2id = {j: i for i, j in self.id2label.items()}  # 建立类别名到id的映射
        self.num_labels = len(self.labels) * 2 + 1  # 13 * 2 + 1 = 27

    def load_data(self, filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in f.split('\n\n'):  # 划分句子
                if not l:
                    continue
                d, last_flag = [], ''
                for c in l.split('\n'):  # 划分字符
                    try:
                        char, this_flag = c.split('\t')
                    except:
                        print(c)
                        continue
                    if this_flag == 'O' and last_flag == 'O':
                        d[-1][0] += char
                    elif this_flag == 'O' and last_flag != 'O':
                        d.append([char, 'O'])
                    elif this_flag[:1] == 'B':
                        d.append([char, this_flag[2:]])
                    else:
                        d[-1][0] += char
                    last_flag = this_flag
                D.append(d)  # 将字符根据BOI拼接成实体保存至列表
        return D

