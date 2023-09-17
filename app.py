import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path

from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten #action detectionimport tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

import pygame 
from pygame import mixer


config = 'config/config.json'



with open(config,'r',encoding="utf-8") as f:
    jsss = json.load(f)

musicpath = jsss['musicPath']   #音乐路径
yllw = jsss['music']['yllw']['musiclist']   #幽灵猎物
zjls = jsss['music']['zjls']['musiclist']   #终极猎手
skls = jsss['music']['skls']['musiclist']   #时空猎手
sqls = jsss['music']['sqls']['musiclist']   #圣泉猎手
jsls = jsss['music']['jsls']['musiclist']   #救赎猎手
jxls = jsss['music']['jxls']['musiclist']   #机械猎手

# allsonglist = [yllw,zjls,skls,sqls,jsls,jxls]
#dict
allsonglist = {'yllw':yllw,'zjls':zjls,'skls':skls,'sqls':sqls,'jsls':jsls,'jxls':jxls}

def CheckfileIsExist():
    for i in allsonglist:
        for j in allsonglist[i]:
            if not os.path.exists(musicpath+j):
                print('文件不存在：'+j)
                exit()
    print('文件检查完毕，全部存在')


IMAGE_SIZE = 128

pygame.init()



def MusicPlayer(hunter):
    music = allsonglist[hunter][np.random.randint(0,len(allsonglist[hunter]))]
    player = musicpath+music
    print('正在播放：'+player)
    mixer.music.load(musicpath+music)
    
    mixer.music.set_volume(0.2)
    mixer.music.play()



class_names = ['jsls', 'jxls', 'other', 'skls', 'sqls', 'yllw', 'zjls']

model = load_model('model.h5')
model.summary()

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    img_array = tf.image.resize(img_array, [IMAGE_SIZE, IMAGE_SIZE])

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence



#全局变量，防止重复播放
last_result = None

i = 0

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


def CameraCheck():
    global i
    global last_result
    while True:
        ret, frame = video.read()
        if not ret:
            continue
        i+=1
        if i % 10 != 0:
            continue
        frame = frame[970:1080, 0:170]
        img = frame
        result = predict(model, img)
        if last_result != result[0]:
            last_result = result[0]
            if result[0] == 'other':
                mixer.music.stop()
            else:
                MusicPlayer(result[0])

        cv2.putText(img, result[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(result[1]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('img', img)
        #按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            

if __name__ == '__main__':
    CheckfileIsExist()
    CameraCheck()
    cv2.destroyAllWindows()