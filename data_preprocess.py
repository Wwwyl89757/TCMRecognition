import glob
from sklearn.model_selection import train_test_split, KFold
import os
import codecs
import shutil
import numpy as np
from nlpcda import Ner
import tcm

augument = True
labels = tcm.TCM().labels
text_length = 500
file_list = glob.glob('./round1_train/train/*.txt')

kf = KFold(n_splits=5, shuffle=True, random_state=666).split(file_list)
file_list = np.array(file_list)

# # 划分训练集和验证集
# train_filelist, val_filelist = train_test_split(file_list,test_size=0.2,random_state=222)

# get_ipython().system('mkdir  ./round1_train/train_new/')
# get_ipython().system('mkdir ./round1_train/val_new/')


def _cut(sentence):
    """
    将一段文本切分成多个句子
    :param sentence:
    :return:
    """
    new_sentence = []
    sen = []
    for i in sentence:
        if i in ['。', '！', '？', '?'] and len(sen) != 0:
            sen.append(i)
            new_sentence.append("".join(sen))
            sen = []
            continue
        sen.append(i)

    if len(new_sentence) <= 1:  # 一句话超过max_seq_length且没有句号的，用","分割，再长的不考虑了。
        new_sentence = []
        sen = []
        for i in sentence:
            if i.split(' ')[0] in ['，', ','] and len(sen) != 0:
                sen.append(i)
                new_sentence.append("".join(sen))
                sen = []
                continue
            sen.append(i)
    if len(sen) > 0:  # 若最后一句话无结尾标点，则加入这句话
        new_sentence.append("".join(sen))
    return new_sentence


def cut_test_set(text_list, len_treshold):
    cut_text_list = []
    cut_index_list = []
    for text in text_list:

        temp_cut_text_list = []
        text_agg = ''
        if len(text) < len_treshold:
            temp_cut_text_list.append(text)
        else:
            sentence_list = _cut(text)  # 一条数据被切分成多句话
            for sentence in sentence_list:
                if len(text_agg) + len(sentence) < len_treshold:
                    text_agg += sentence
                else:
                    temp_cut_text_list.append(text_agg)
                    text_agg = sentence
            temp_cut_text_list.append(text_agg)  # 加上最后一个句子

        cut_index_list.append(len(temp_cut_text_list))
        cut_text_list += temp_cut_text_list

    return cut_text_list, cut_index_list


# # 数据处理

#设置样本长度

def from_ann2dic(r_ann_path, r_txt_path, w_path, w_file):
    q_dic = {}
    with codecs.open(r_ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n\r")               # 去除标签数据每行首尾空格
            line_arr = line.split('\t')             # 按tab分割成id，(类别，起始位置，结束位置)，实体
            entityinfo = line_arr[1]
            entityinfo = entityinfo.split(' ')
            cls = entityinfo[0]
            start_index = int(entityinfo[1])
            end_index = int(entityinfo[2])
            length = end_index - start_index
            for r in range(length):
                if r == 0:
                    q_dic[start_index] = ("B-%s" % cls)         # 实体起始位置标为 B-类别
                else:
                    q_dic[start_index + r] = ("I-%s" % cls)     # 实体中部标为 I-类别

    with codecs.open(r_txt_path, "r", encoding="utf-8") as f:
        content_str = f.read()
        
    
    cut_text_list, cut_index_list = cut_test_set([content_str],text_length)
    
    i = 0
    for idx, line in enumerate(cut_text_list):
        w_path_ = "%s/%s-%s-new.txt" % (w_path, w_file, idx)
        with codecs.open(w_path_, "w", encoding="utf-8") as w:
            for str_ in line:
                if str_ is " " or str_ == "" or str_ == "\n" or str_ == "\r":
                    pass
                else:
                    if i in q_dic:
                        tag = q_dic[i]
                    else:
                        tag = "O"  # 大写字母O
                    w.write('%s\t%s\n' % (str_, tag))            # 将.ann中标签拆分成（字符 BIO）写入-new.txt
                i+=1
            # w.write('%s\n' % "END O")

def process():
    for i, (train_fold, val_fold) in enumerate(kf):
        if os.path.exists('./round1_train/train_new_%s/' % i):
            shutil.rmtree('./round1_train/train_new_%s/' % i)

        if os.path.exists('./round1_train/val_new_%s/' % i):
            shutil.rmtree('./round1_train/val_new_%s/' % i)

        os.mkdir("./round1_train/train_new_%s" % i)
        os.mkdir("./round1_train/val_new_%s" % i)

        train_filelist = list(file_list[train_fold])
        val_filelist = list(file_list[val_fold])

        data_dir = './round1_train/train/'
        # # 训练集处理
        for file in train_filelist:
            if file.find(".ann") == -1 and file.find(".txt") == -1:
                continue
            file_name = file.split('\\')[-1].split('.')[0]
            r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
            r_txt_path = os.path.join(data_dir, "%s.txt" % file_name)
            w_path = f'./round1_train/train_new_%s/' % i
            w_file = file_name
            from_ann2dic(r_ann_path, r_txt_path, w_path, w_file)
        # # 验证集处理
        for file in val_filelist:
            if file.find(".ann") == -1 and file.find(".txt") == -1:
                continue
            file_name = file.split('\\')[-1].split('.')[0]
            r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
            r_txt_path = os.path.join(data_dir, "%s.txt" % file_name)
            w_path = './round1_train/val_new_%s/' % i
            w_file = file_name
            from_ann2dic(r_ann_path, r_txt_path, w_path, w_file)
        # # 训练集合并
        if augument:
            ner = Ner(ner_dir_name="./round1_train/train_new_%s" % i, ignore_tag_list=['O'],
                      data_augument_tag_list=labels, augument_size=3, seed=0)
        w_path = "./round1_train/data/train_%s.txt" % i
        with codecs.open(w_path, 'w', encoding='utf-8') as f:
            f.seek(0)  # 移动文件读取指针到指定位置
            f.truncate()  # 写入前先清空之前的文件内容
        for file in os.listdir('./round1_train/train_new_%s/' % i):
            path = os.path.join("./round1_train/train_new_%s" % i, file)
            if not file.endswith(".txt"):
                continue
            q_list = []
            print("开始读取文件:%s" % file)
            with codecs.open(path, "r", encoding="utf-8") as f:
                line = f.readline()
                line = line.strip("\n\r")
                end = 0
                while not end:
                # while line != "END O":
                    q_list.append(line)
                    line = f.readline()
                    if line != '':
                        line = line.strip("\n\r")
                    else:
                        end = 1
                        break
            print("开始写入文本%s" % w_path)
            with codecs.open(w_path, "a", encoding="utf-8") as f:  # 将train_new中的text合并写入data/train，追加写入
                for item in q_list:
                    if item.__contains__('\ufeff1'):
                        print("===============")
                    f.write('%s\n' % item)

                if augument:
                    data_sentence_arrs, data_label_arrs = ner.augment(file_name=path)
                    for j in range(len(data_sentence_arrs)):
                        for str_, tag in zip(data_sentence_arrs[j], data_label_arrs[j]):
                            f.write("%s\t%s\n" % (str_, tag))
                f.write('\n')
            f.close()

        # # 验证集合并
        w_path = "./round1_train/data/val_%s.txt" % i
        with codecs.open(w_path, 'w', encoding='utf-8') as f:
            f.seek(0)  # 移动文件读取指针到指定位置
            f.truncate()  # 写入前先清空之前的文件内容
        for file in os.listdir('./round1_train/val_new_%s/' % i):
            path = os.path.join("./round1_train/val_new_%s" % i, file)
            if not file.endswith(".txt"):
                continue
            q_list = []
            print("开始读取文件:%s" % file)
            with codecs.open(path, "r", encoding="utf-8") as f:
                line = f.readline()
                line = line.strip("\n\r")
                end = 0
                while not end:
                    # while line != "END O":
                    q_list.append(line)
                    line = f.readline()
                    if line != '':
                        line = line.strip("\n\r")
                    else:
                        end = 1
                        break
            print("开始写入文本%s" % w_path)
            with codecs.open(w_path, "a", encoding="utf-8") as f:
                for item in q_list:
                    if item.__contains__('\ufeff1'):
                        print("===============")
                    f.write('%s\n' % item)
                f.write('\n')
            f.close()

    # # 原始验证集拷贝
    if os.path.exists('./round1_train/val/'):
        shutil.rmtree('./round1_train/val/')

    os.mkdir('./round1_train/val/')

    for file in val_filelist:
        file_name = file.split('\\')[-1].split('.')[0]
        r_ann_path = os.path.join("./round1_train/train", "%s.ann" % file_name)
        # os.system("cp %s %s" % (file, "./round1_train/val_data"))               # 将验证集对应.txt文件复制到val_data
        # os.system("cp %s %s" % (r_ann_path, "./round1_train/val_data"))         # 将验证集对应.ann文件复制到val_data
        shutil.copy(file, "./round1_train/val")
        shutil.copy(r_ann_path, "./round1_train/val")
        # print(file)


if __name__ =='__main__':
    process()