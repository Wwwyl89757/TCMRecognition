from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator

class Generator(DataGenerator):
    """数据生成器
       """

    def __init__(self, train_data, batch_size, tokenizer, maxlen, label2id):
        super(Generator, self).__init__(data=train_data, batch_size=batch_size)
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.label2id = label2id

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [self.tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = self.tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < self.maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = self.label2id[l] * 2 + 1
                        I = self.label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [self.tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

