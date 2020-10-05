import keras
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from tqdm import tqdm
from models import NamedEntityRecognizer
from bert4keras.layers import ConditionalRandomField
from tcm import TCM


class Evaluator(keras.callbacks.Callback):
    def __init__(self, valid_data, tokenizer: Tokenizer, model, NER: NamedEntityRecognizer, CRF: ConditionalRandomField, loader: TCM):
        self.best_val_f1 = 0
        self.valid_data = valid_data
        self.tokenizer = tokenizer
        self.model = model
        self.NER = NER
        self.CRF = CRF
        self.loader = loader

    def evaluate(self):
        """评测函数
        """
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for d in tqdm(self.valid_data):
            text = ''.join([i[0] for i in d])  # 将列表拼接成完整的句子
            R = set(self.NER.recognize(text, self.tokenizer, self.model, self.loader))  # 预测，实体和类别名的集合
            T = set([tuple(i) for i in d if i[1] != 'O'])  # 真实
            X += len(R & T)  # X为准确预测的数量
            Y += len(R)  # Y为预测的总量
            Z += len(T)  # Z为真实实体数量
        precision, recall = X / Y, X / Z
        f1 = 2 * precision * recall / (precision + recall)
        return f1, precision, recall

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(self.CRF.trans)
        self.NER.trans = trans
#         print(NER.trans)
        f1, precision, recall = self.evaluate()
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.model.save_weights('./best_model_epoch_10.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )