
from data_utils import ENTITIES, Documents, Dataset, SentenceExtractor, make_predictions
from models import build_lstm_crf_model
import numpy as np

ent2idx = dict(zip(ENTITIES, range(1, len(ENTITIES) + 1)))
idx2ent = dict([(v, k) for k, v in ent2idx.items()])
num_cates = max(ent2idx.values()) + 1
sent_len = 64
vocab_size = 2320
emb_size = 256
sent_pad = 10
seq_len = sent_len + 2 * sent_pad

test_data_dir = '../data/chusai_xuanshou'
test_docs = Documents(data_dir=test_data_dir)
sent_extrator = SentenceExtractor(window_size=sent_len, pad_size=sent_pad)
test_sents = sent_extrator(test_docs)

with open('word2idx.json', 'r') as f:
    word2idx = eval(f.read())

test_data = Dataset(test_sents, word2idx=word2idx, cate2idx=ent2idx)
test_X, _ = test_data[:]

print(len(test_docs))

w2v_embeddings = np.load('w2v_embeddings.npy')

model = build_lstm_crf_model(num_cates, seq_len=seq_len, vocab_size=vocab_size,
                             model_opts={'emb_matrix': w2v_embeddings, 'emb_size': emb_size, 'emb_trainable': False})
model.load_weights('../model_files/BiLSTM-CRF_epochs10_sentlen64_vocabsize2320_embsize256_sentpad10_fscore0.7599364069952306_precision0.7810457516339869_recall0.739938080495356.weights')
print(model.summary())

preds = model.predict(test_X, batch_size=64, verbose=True)
print(preds)