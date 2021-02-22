# -*- coding: utf-8 -*-
# @Author  : zzy
# @FileName: collectData.py
# @Time    : 2020/12/15 11:06

import cv2
import os
from faceRecognition import detectFace, drawRectangle

"""
保存图片
"""

dataFoderPath = "trainData"
name = input("Please input your name:")

label = len(os.listdir(dataFoderPath))
imagePath = dataFoderPath + "/" + str(label) + "_" + name
for data in os.listdir(dataFoderPath):
    n = data.split("_")[1]
    if n == name:
        imagePath = dataFoderPath + "/" + data
        label = int(data.split("_")[0])
        break

if not os.path.exists(imagePath):
    os.makedirs(imagePath)

capture = cv2.VideoCapture(0)
count = len(os.listdir(imagePath))

while True:
    try:
        ret, frame = capture.read()
        face, rect = detectFace(frame)
        frameCopy = frame.copy()
        if rect is not None:
            drawRectangle(frameCopy, rect)
        cv2.imshow("data", frameCopy)
        k = cv2.waitKey(1)
        if k == 27:  # 通过esc键退出摄像
            break
        if k == 32:
            cv2.imwrite(imagePath + "/" + str(label) + "_" + str(count) + '.jpg', frame)
            count = count + 1
    except :
        cv2.destroyAllWindows()
        capture.release()
        break
