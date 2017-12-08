# -*- coding: utf-8 -*-
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import _pickle as pk
import sys

def loadData_test(filePath):
    count=0
    test_List=[]
    with open(filePath, encoding='utf-8') as f : 
        lines = f.readlines()
        for line in lines:
            if count != 0:
                split_line = line.strip().split(',', 1)
                test_List.append(split_line[1])
            count = count + 1
    f.close()
    return test_List

print("======load test data========")
test_list_data = loadData_test(sys.argv[1])
print("======load test data Done========")
tokenizer = pk.load(open("data/tokenizer.pk", 'rb'))
max_len = 30
test_list_seq = tokenizer.texts_to_sequences(test_list_data)
x_test = sequence.pad_sequences(test_list_seq, maxlen=max_len)

np.save('data/x_test.npy', x_test)
print("-----DONE-----")
