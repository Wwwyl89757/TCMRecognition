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

    def _cut(self, sentence):
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

    def cut_test_set(self, text_list, len_treshold):
        cut_text_list = []
        cut_index_list = []
        for text in text_list:

            temp_cut_text_list = []
            text_agg = ''
            if len(text) < len_treshold:
                temp_cut_text_list.append(text)
            else:
                sentence_list = self._cut(text)  # 一条数据被切分成多句话
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