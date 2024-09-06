# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:15:08 2023

@author: nagal
"""

print("********************************************************************************************************************")
print("Import libraries")
print("********************************************************************************************************************")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#************************************************************************
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier as knn 
from sklearn.feature_extraction.text import CountVectorizer
#************************************************************************
import nltk
#nltk.download('punkt')  # Run atleast once  with mobile net not with JIO
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
#nltk.download('averaged_perceptron_tagger') # Run atleast once 
from nltk import FreqDist
#************************************************************************



from nltk.tokenize import RegexpTokenizer

print("Load Dataset")
data=pd.read_csv('fash_reviews.csv')

print("d.A")
print("Data Head is :\n",data.head(3))
print("************")
print("Data Columns are :\n",data.columns)
print("************")
print("Data Shape is:\n",data.shape)
print("************")
print("Data Description is :",data.describe())
print("************")
print("Classes Info is :",data.groupby("Division Name").size())
