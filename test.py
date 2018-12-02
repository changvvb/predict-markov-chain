import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from string import punctuation
import pandas as pd
import yaml
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import sys
sys.setrecursionlimit(1000000)

# define parameters
maxlen = 100

def create_dictionaries(model=None,
        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined): # 闭包-->临时使用
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) # freqxiao10->0
                data.append(new_txt)
            return data # word=>index
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print('No data provided...')


def input_transform(combined):
    model=Word2Vec.load('./Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,combined)
    return combined


def lstm_predict(string):
    with open('./lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    model.load_weights('./lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)
    #print data
    result=model.predict_classes(data)
    proba_result = model.predict_proba(data)
    
    print(proba_result)
    proba_result = [v[1] for v in proba_result]

    return result,proba_result


if __name__=='__main__':
    df = pd.read_csv("./question1_sentiment.csv")
    df = df.tail(5000)
    punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'

    origin_df = df[['id','review']]
    df['review'] = df['review'].replace("""<br /><br />""",' ') \
            .replace(r'\\',' ',regex=True) \
            .replace(r'[{}]'.format(punc),' ',regex=True) \
            .replace(r' +',' ',regex=True) \
            .replace(r'[^\x00-\x7F]+',' ',regex=True).str.lower()

    sentences = df['review'].str.split()

    #  inputs = input_transform(sentences)

    #  print(inputs)

    result,proba_result = lstm_predict(sentences)

    origin_df['result'] = result
    origin_df['sentiment'] = proba_result


    print(origin_df)
    origin_df.to_csv("./2018180138.csv",index=False)



