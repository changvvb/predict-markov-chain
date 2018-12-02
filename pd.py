import pandas as pd
from gensim.models.word2vec import Word2Vec
from string import punctuation

window_size = 7
n_exposures = 10 # 所有频数超过10的词语

#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    model = Word2Vec(min_count=n_exposures,
                     window=window_size)
    model.build_vocab(combined) # input: list
    model.train(combined,total_examples=model.corpus_count, epochs=model.iter)
    model.save('./Word2vec_model.pkl')
    model.wv.save_word2vec_format("./word2vec.model",binary=True)



df = pd.read_csv("./question1_sentiment.csv",nrows=20000)
punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'

df['review'] = df['review'].str.lower()
df['review'] = df['review'].replace(r'(<br />)',' ',regex=True) \
        .replace(r'\\',' ',regex=True) \
        .replace(r'[{}]'.format(punc),' ',regex=True) \
        .replace(r'(?:^|(?<= ))(for|a|of|the|and|to|in)(?:(?= )|$)',' ',regex=True) \
        .replace(r' +',' ',regex=True) \
        .replace(r'[^\x00-\x7F]+',' ',regex=True)

sentences = df['review'].str.split()

print(sentences)
word2vec_train(sentences)
