#!/usr/bin/python 
# -*- coding: utf-8 -*-

import os
import json
import jieba


dataPath = 'original-microblog/'


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


def main():
    infoList = allFileInfo(dataPath)

    for fileInfo in infoList: # fileInfo[0]: filename, fileInfo[1]: fileTag
        file = open(fileInfo[0], encoding='utf-8', mode='r')
        data = file.read()
        jsonObj = json.loads(data)
        text = jsonObj['text']
        
        '''
            ...
        '''

        file.close()


main()
