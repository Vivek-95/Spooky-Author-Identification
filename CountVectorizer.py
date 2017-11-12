# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:16:44 2017

@author: Pramod
"""

import pandas as pd
#import re
#import numpy as np
#from nltk.corpus import stopwords
#from nltk.stem import SnowballStemmer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.svm import LinearSVC
#from sklearn.pipeline import Pipeline
##from sklearn.model_selection import train_test_split
#from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

text_data = pd.read_csv("D:/Datasets/Spooky Author Identification/train.csv")
#print(text_data.head())
#print(text_data.shape)
#print(text_data.author.value_counts())
text_data["author_label"] = text_data.author.map({'EAP':0,'HPL':1,'MWS':2})
#print(text_data.shape)
#print(text_data.head())

X_train = text_data.text
y_train = text_data.author_label
vec = CountVectorizer(stop_words='english')
vec.fit(X_train)
train_matrix = vec.transform(X_train)
train_df = pd.DataFrame(train_matrix.toarray(), columns=vec.get_feature_names())

mnb = MultinomialNB()
mnb.fit(train_df,y_train)

test_data = pd.read_csv("D:/Datasets/Spooky Author Identification/test.csv")
#print(test_data.shape)
X_test = test_data.text
vec = CountVectorizer(stop_words='english')
vec.fit(X_test)
test_matrix = vec.transform(X_test)
test_df = pd.DataFrame(test_matrix.toarray(), columns=vec.get_feature_names())

test_y = mnb.predict(test_df)
#
#from sklearn.linear_model import LogisticRegression
#logreg = LogisticRegression()
#
#
## train the model using X_train_dtm
#logreg.fit(train_df, y_train)
#
#
## make class predictions for X_test_dtm
#test_y = logreg.predict(test_df)
#
final_df = test_df.id
final_df = test_y

final_df.to_csv(path = "C:/Users/user/Desktop")
