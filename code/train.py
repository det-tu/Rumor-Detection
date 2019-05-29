import torch
import numpy as np
import math
import LSTM
import os
import json
import jieba
import gensim
import random

dataPath = '../dataset/dataset/train/'
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
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

infoList = allFileInfo(dataPath)

model = gensim.models.Word2Vec.load("version2.model.bin")

for epoch in range(10):
    for fileCount in range(len(infoList)):
        rand = random.randint(0,len(infoList)-1)
        fileInfo = infoList[rand]
        del infoList[rand]
        '''
        for fileInfo in infoList: # fileInfo[0]: filename, fileInfo[1]: fileTag
        '''
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

        if (batch_size == 0):
            continue
        x = torch.zeros(batch_size,input_size)
        y = torch.zeros(1)
        y[0] = fileInfo[1]
        try:
            row = 0
            for word in rawList:
                num = torch.from_numpy(model[word])
                x[row] = num
                row+=1
        except:
            continue

        x_form = x.view(1,batch_size,-1)
        #x_form = x[:, np.newaxis]
    
        prediction = net(x_form)
        #predict = max(prediction[0],prediction[1])
        #prediction = prediction.view(1,2)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(net.state_dict(),'weights.pkl')