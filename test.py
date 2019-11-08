import Terry_toolkit as tkit
import jieba
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def test(text=''):
    model_dm = Doc2Vec.load("model/doc2vec.model")
    text_cut = jieba.cut(text)
    text_raw = []
    for i in list(text_cut):
        text_raw.append(i)
    inferred_vector_dm = model_dm.infer_vector(text_raw)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    return sims



sims = test('linux操作系统真慢')
# print(sims)

for item in sims:
    print(item)
# for count, sim in sims:
# sentence = x_train[count]
# words = ''
# for word in sentence[0]:
#     words = words + word + ' '
# print(words, sim, len(sentence[0]))