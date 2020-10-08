from bert4keras.snippets import open, ViterbiDecoder, to_array
import tcm
from keras.models import Model

class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text, tokenizer, models, loader):
        tokens = tokenizer.tokenize(text)           # 将文本切分成字符
        mapping = tokenizer.rematch(text, tokens)   # 将字符按顺序映射成id
        token_ids = tokenizer.tokens_to_ids(tokens)  # 将字符按字典映射成id
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = 0
        if isinstance(models, list):
            for model in models:
                nodes += model.predict([token_ids, segment_ids])[0]  # shape[len(text), 27]
            nodes /= len(models)
        else:
            nodes = models.predict([token_ids, segment_ids])[0]  # shape[len(text), 27]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:              # 如果标签id为奇数则为实体起始位置
                    starting = True
                    entities.append([[i], loader.id2label[(label - 1) // 2]])  # 根据起始位置的label确定实体类别
                elif starting:
                    entities[-1][0].append(i)  # 实体内部字符的下标加入列表，第二个维度只有一个值为类别名称，所以都取0
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]