#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import keras
import sys
import bert4keras
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
import glob
import codecs
from tqdm import tqdm
import generator
import models
import evaluator
import tcm


maxlen = 250
epochs = 5
batch_size = 16
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

# bert配置
config_path = './publish/bert_config.json'
checkpoint_path = './publish/bert_model.ckpt'
dict_path = './publish/vocab.txt'

# print(tf.test.is_gpu_available())
# print(tf.config.list_physical_devices('GPU'))
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)  # CRF层本质上是一个带训练参数的loss计算层
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# 标注数据
loader = tcm.TCM()


def train():
    train_data = loader.load_data('./round1_train/data/train.txt')  # 三维列表，第一个维度为句子个数，另外是实体和对应类别
    valid_data = loader.load_data('./round1_train/data/val.txt')

    global train_generator
    train_generator = generator.Generator(train_data=train_data, batch_size=batch_size,
                                          tokenizer=tokenizer, maxlen=maxlen, label2id=loader.label2id)

    global model
    model = build_transformer_model(
        config_path,
        checkpoint_path,
    )                                                   # 根据bert_model.ckpt和bert_config.json文件构建transformer模型

    output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)  # 'Transformer-11-FeedForward-Norm'
    output = model.get_layer(output_layer).output  # Tensor("Transformer-11-FeedForward-Norm/add_1:0", shape=(None, None, 768), dtype=float32)
    output = Dense(loader.num_labels)(output)      # 27分类，Tensor("dense_73/add:0", shape=(None, None, 27), dtype=float32)

    output = CRF(output)

    model = Model(model.input, output)
    model.summary()

    model.compile(
        loss=CRF.sparse_loss,
        optimizer=Adam(learing_rate),
        metrics=[CRF.sparse_accuracy]
    )

    NER = models.NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
    evaluate = evaluator.Evaluator(valid_data, tokenizer, model, NER, CRF, loader)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluate]
    )


def verify():
    NER = models.NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
    # 验证集
    X, Y, Z = 1e-10, 1e-10, 1e-10
    val_data_flist = glob.glob('./round1_train/val_data/*.txt')
    data_dir = './round1_train/val_data/'
    for file in val_data_flist:
        if file.find(".ann") == -1 and file.find(".txt") == -1:
            continue
        file_name = file.split('\\')[-1].split('.')[0]
        r_ann_path = os.path.join(data_dir, "%s.ann" % file_name)
        r_txt_path = os.path.join(data_dir, "%s.txt" % file_name)

        R = []
        with codecs.open(r_txt_path, "r", encoding="utf-8") as f:
            line = f.readlines()
            aa = predict_test(line, NER)
            for line in aa[0]:
                lines = line['label_type'] + " " + str(line['start_pos']) + ' ' + str(line['end_pos']) + "\t" + line[
                    'res']
                R.append(lines)
        T = []
        with codecs.open(r_ann_path, "r", encoding="utf-8") as f:
            for line in f:
                lines = line.strip('\n').split('\t')[1] + '\t' + line.strip('\n').split('\t')[2]
                T.append(lines)
        R = set(R)
        T = set(T)
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    precision, recall = X / Y, X / Z
    f1 = 2 * precision * recall / (precision + recall)


def tcm_test():
    # ## 测试集
    NER = models.NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

    test_files = os.listdir("./round1_test/chusai_xuanshou/")

    for file in test_files:
        with codecs.open("./round1_test/chusai_xuanshou/" + file, "r", encoding="utf-8") as f:
            line = f.readlines()
            aa = predict_test(line, NER)
        with codecs.open("./round1_test/submission_4/" + file.split('.')[0] + ".ann", "w", encoding="utf-8") as ff:
            for line in aa[0]:
                lines = line['overlap'] + "\t" + line['label_type'] + " " + str(line['start_pos']) + ' ' + str(
                    line['end_pos']) + "\t" + line['res']
                ff.write(lines + "\n")
            ff.close()


def predict_test(data, NER_):
    test_ner =[]

    for text in tqdm(data):
        cut_text_list, cut_index_list = train_generator.cut_test_set([text], maxlen)  # 将文本分成多个句子
        posit = 0
        item_ner = []
        index = 1
        for str_ in cut_text_list:
            aaaa = NER_.recognize(str_, tokenizer, model, loader)
            for tn in aaaa:
                ans = {}
                ans["label_type"] = tn[1]
                ans['overlap'] = "T" + str(index)

                ans["start_pos"] = text.find(tn[0], posit)
                ans["end_pos"] = ans["start_pos"] + len(tn[0])
                posit = ans["end_pos"]
                ans["res"] = tn[0]
                item_ner.append(ans)
                index += 1
        test_ner.append(item_ner)

    return test_ner


if __name__ =='__main__':
    train()
    verify()
    tcm_test()