import jiebaimport jieba.posseg as psegfrom gensim.models import Word2Vecs = "我爱中国"l = jieba._lcut(s)print(l)words = pseg.cut(s)result = []for w in words:    result.append(str(w.word)+"/"+str(w.flag))print(result)