import numpy as np
from sklearn.model_selection import ShuffleSplit
from data_utils import ENTITIES, Documents, Dataset, SentenceExtractor, make_predictions
from data_utils import Evaluator
from models import build_lstm_crf_model
from gensim.models import Word2Vec


# read data
data_dir = '../data/train'
ent2idx = dict(zip(ENTITIES, range(1, len(ENTITIES) + 1)))
idx2ent = dict([(v, k) for k, v in ent2idx.items()])

docs = Documents(data_dir=data_dir)
rs = ShuffleSplit(n_splits=1, test_size=20, random_state=2018)
train_doc_ids, test_doc_ids = next(rs.split(docs))
train_docs, test_docs = docs[train_doc_ids], docs[test_doc_ids]
print(len(docs))

epochs = 10
num_cates = max(ent2idx.values()) + 1
sent_len = 64
vocab_size = 3000
emb_size = 256
sent_pad = 10
sent_extrator = SentenceExtractor(window_size=sent_len, pad_size=sent_pad)
train_sents = sent_extrator(train_docs)
test_sents = sent_extrator(test_docs)

train_data = Dataset(train_sents, cate2idx=ent2idx)
train_data.build_vocab_dict(vocab_size=vocab_size)

with open('word2idx.json', 'w') as f:
    f.write(str(train_data.word2idx))

test_data = Dataset(test_sents, word2idx=train_data.word2idx, cate2idx=ent2idx)
test_X, _ = test_data[:]

vocab_size = len(train_data.word2idx)

w2v_train_sents = []
for doc in docs:
    w2v_train_sents.append(list(doc.text))
w2v_model = Word2Vec(w2v_train_sents, size=emb_size)

w2v_embeddings = np.zeros((vocab_size, emb_size))
for char, char_idx in train_data.word2idx.items():
    if char in w2v_model.wv:
        w2v_embeddings[char_idx] = w2v_model.wv[char]

np.save("w2v_embeddings.npy", w2v_embeddings)

seq_len = sent_len + 2 * sent_pad
model = build_lstm_crf_model(num_cates, seq_len=seq_len, vocab_size=vocab_size,
                             model_opts={'emb_matrix': w2v_embeddings, 'emb_size': emb_size, 'emb_trainable': False})
print(model.summary())

train_X, train_y = train_data[:]
print('train_X.shape', train_X.shape)
print('train_y.shape', train_y.shape)

model.fit(train_X, train_y, batch_size=64, epochs=epochs)

# make prediction
preds = model.predict(test_X, batch_size=64, verbose=True)
pred_docs = make_predictions(preds, test_data, sent_pad, docs, idx2ent)

# evaluate on validatation set
f_score, precision, recall = Evaluator.f1_score(test_docs, pred_docs)
print('f_score: ', f_score)
print('precision: ', precision)
print('recall: ', recall)

model.save_weights('../model_files/BiLSTM-CRF_epochs{}_sentlen{}_vocabsize{}_embsize{}_sentpad{}'
           '_fscore{}_precision{}_recall{}.weights'.format(
    epochs, sent_len, vocab_size, emb_size, sent_pad, f_score, precision, recall))
