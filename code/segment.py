import jieba
import jieba.analyse
from gensim.models import word2vec
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.LineSentence('./wordbase_new.txt')
model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100)
print(type(sentences))

model.save("version2.model.bin")