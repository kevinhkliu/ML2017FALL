# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 
from sklearn.cross_validation import train_test_split

c = pd.read_csv('data/train.csv')
user_list = c['UserID'].values
movie_list = c['MovieID'].values
rating_list = c['Rating'].values

user_train, user_val, movie_train, movie_val, rating_train, rating_val= train_test_split(user_list, movie_list, rating_list, test_size=0.05, random_state=42)
print("===save data=====")
np.save('data/user_train.npy', user_train)
np.save('data/user_val.npy', user_val)
np.save('data/movie_train.npy', movie_train)
np.save('data/movie_val.npy', movie_val)
np.save('data/rating_train.npy', rating_train)
np.save('data/rating_val.npy', rating_val)
print("===save data Done=====")
