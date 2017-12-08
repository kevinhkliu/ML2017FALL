# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
import gensim
import _pickle as pk
import sys

def loadData_train(filePath):
    twitter_List=[]
    label_List=[]
    count=0
    with open(filePath, encoding='utf-8') as f : 
        lines = f.readlines()
        for line in lines:
            split_line = line.strip().split(" +++$+++ ")
            label_List.append(split_line[0])
            twitter_List.append(split_line[1])
            count = count + 1
    f.close()
    return twitter_List, label_List

def loadData_train_nolabel(filePath):
    twitter_List_nolabel=[]
    with open(filePath, encoding='utf-8') as f : 
        lines = f.readlines()
        for line in lines:
            twitter_List_nolabel.append(line.strip())
    f.close()
    return twitter_List_nolabel

def process_doc_list(doc_list_data):
    doc_corpus = []
    doc_list_str = []
    for doc_str_data in doc_list_data:
        doc_tokens_data = text_to_word_sequence(doc_str_data,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")
        doc_tokens = []
        doc_str = ''
        for word in doc_tokens_data:
            doc_tokens.append(word)
            doc_str = doc_str + ' ' + word
        doc_corpus.append(doc_tokens)
        doc_list_str.append(doc_str)

    return doc_corpus, doc_list_str

print("======load train data========")
twitter_list_data, label_list = loadData_train(sys.argv[1])
print("======load train data Done========")

print("======load test data nolabel========")
twitter_list_data_nolabel = loadData_train_nolabel(sys.argv[2])
print("======load test data nolabel Done========")

twitter_corpus, twitter_list = process_doc_list(twitter_list_data)
twitter_corpus_nolabel, twitter_list_nolabel = process_doc_list(twitter_list_data_nolabel)

max_len = 30

total_twitter_list = twitter_list + twitter_list_nolabel
total_twitter_corpus = twitter_corpus + twitter_corpus_nolabel

label_len = len(twitter_list)

tokenizer = Tokenizer(num_words=80000,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ",char_level=False)
tokenizer.fit_on_texts(total_twitter_list)

twitter_list_seq = tokenizer.texts_to_sequences(twitter_list)
x_train = sequence.pad_sequences(twitter_list_seq, maxlen=30)

twitter_list_nolabel_seq = tokenizer.texts_to_sequences(twitter_list_data_nolabel)
x_train_nolabel = sequence.pad_sequences(twitter_list_nolabel_seq, maxlen=max_len)


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

MAX_NB_WORDS = 80000

embedding_size = 300    # Word vector dimensionality                      
min_word_count = 10   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 5          # Context window    size                                                                                    

print("train gensim word2vector")
w2vModel = gensim.models.Word2Vec(total_twitter_corpus, size=embedding_size, window=context,
                                  min_count=min_word_count, workers=num_workers)
#                                  sample = downsampling)
#w2vModel.save('save/trained.model')
# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, embedding_size))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    if word in w2vModel.wv.vocab:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = w2vModel.wv[word]

x_train, x_val, y_train, y_val = train_test_split(x_train, label_list, test_size=0.2, random_state=42)
   
print("save data model")
np.save('data/gensim_word2vec', embedding_matrix)
np.save('data/x_train_nolabel.npy', x_train_nolabel)
np.save('data/x_train.npy', x_train)
np.save('data/x_val.npy', x_val)
np.save('data/y_train.npy', y_train)
np.save('data/y_val.npy', y_val)
pk.dump(tokenizer, open("data/tokenizer.pk", 'wb'))
