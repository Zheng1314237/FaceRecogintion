# -*- coding: utf-8 -*-
# @Author  : zzy
# @FileName: main.py
# @Time    : 2020/12/15 11:07

import faceRecognition
import os
import cv2
import numpy as np
import re

choose = 1
capture = cv2.VideoCapture(0)
names = {}
for datapath in os.listdir("trainData"):
    name = datapath.split("_")[1]
    label = datapath.split("_")[0]
    names[label] = name
try:
    train = int(input("please choose train (input 1) or test (input 2):"))
except:
    print("choose error")

while True:
    try:
        faceRecoginzer = (cv2.face.LBPHFaceRecognizer_create() if (choose == 1) else (
            cv2.face.FisherFaceRecognizer_create() if (choose == 2) else cv2.face.EigenFaceRecognizer_create()))
        if train == 2 and not os.path.exists("trainModel/train.xml"):
            print("please train first")
        elif train == 2:
            ret, frame = capture.read()
            faceRecoginzer.read("trainModel/train.xml")
            label, confidence, predictImage1 = faceRecognition.predict(frame, faceRecoginzer,names)
            cv2.namedWindow("predict", cv2.WINDOW_NORMAL)
            cv2.imshow("predict", predictImage1)
            k = cv2.waitKey(1)
            if k == 27:  # 通过esc键退出摄像
                break
        elif train == 1:
            print("Preparing data...")
            faces, labels = faceRecognition.prepareTrainingData("trainData")
            print("Data prepared...")
            faceRecoginzer.train(faces, np.array(labels))
            faceRecoginzer.write(r"trainModel/train.xml")
            print("train complete...")
            break
    except:
        break
