from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text
from sklearn import decomposition, ensemble
from sklearn.naive_bayes import MultinomialNB
from keras import layers, models, optimizers
from gensim.models import Word2Vec

import pandas as pd
import numpy as np
import string, jieba, re, os, sys
import logging

# Extract content of truth



# 多线程分词
jieba.enable_parallel()

# 停用词
def getStopwords():
    stopwords = []
