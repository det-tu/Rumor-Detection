import torch
import numpy as np
import math
import LSTM
import os
import json
import jieba
import gensim
import random


dataPath = '../dataset/dataset/test/'
stopListPath = 'stopWord.txt'


def allFileInfo(filepath):
    infoList = []
    pathDir =  os.listdir(filepath)
    for fileDir in pathDir:
        num = int(fileDir.split('_')[0])
        if num>2600:
            tag = 1
        else:
            tag = 0
        # tag = 1 -> non-rumour
        # tag = 0 -> rumour 
        child = os.path.join('%s%s' % (filepath, fileDir))
        infoList.append([child,tag])

    return infoList


def stopwordslist():
    stopwords = [line.strip() for line in open(stopListPath, encoding='utf-8', mode='r').readlines()]

    return stopwords


input_size = 100
hidden_size = 100
output_size = 1

net = LSTM.LSTM(input_size, hidden_size, output_size)
net.load_state_dict(torch.load('weights.pkl'))

infoList = allFileInfo(dataPath)

model = gensim.models.Word2Vec.load("version2.model.bin")

total = 0
correct = 0

for fileCount in range(len(infoList)):
    rand = random.randint(0,len(infoList)-1)
    fileInfo = infoList[rand]
    del infoList[rand]

    file = open(fileInfo[0], encoding='utf-8', mode='r')
    data = file.read()
    jsonObj = json.loads(data)
    text = jsonObj['text']
    file.close()

    text = text.replace(' ','')
    segList = jieba.lcut(text)
    rawList = []
    stopwords = stopwordslist()
    for word in segList:
        if word not in stopwords:
            rawList.append(word)

    batch_size = len(rawList)

    x = torch.zeros(batch_size,input_size)
    y = torch.zeros(1)
    y[0] = fileInfo[1]
    try:
        row = 0
        for word in rawList:
            num = torch.from_numpy(model[word])
            x[row] = num
            row += 1
    except:
        continue

    x_form = x.view(1,batch_size,-1)
    
    prediction = net(x_form)
    #print(prediction.data.numpy(), fileInfo[1])

    if prediction.data.numpy()[0]>=0.5:
        result = 1
    else:
        result = 0

    total += 1
    if result == fileInfo[1]:
        correct += 1

correct_rate = 1.0*correct/total
print('test complete. total:'+str(total)+', correct:'+str(correct)+', correct rate:'+str(correct_rate))