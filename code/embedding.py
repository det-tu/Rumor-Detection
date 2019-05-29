from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models import Word2Vec

pyplot.rcParams['font.sans-serif']=['SimHei']
pyplot.rcParams['axes.unicode_minus'] = False

model = Word2Vec.load("version2.model.bin")
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
#print(model["天气"].size)

# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

req_count = 5
for key in model.wv.similar_by_word('女朋友', topn =100):
    if len(key[0])==3:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break