#  faceRecogintion

#### 介绍

使用opencv内置的人脸识别模型进行训练的小demo

#### 软件架构

Opencv3

Python3.7

opencv-contrib（opencv扩展包，需要自己安装）

#### 使用说明

1.运行collectData.py 

​    首先会要求填写收集者的名字

​	按下空格键就能保存当前截图，并且按照(label_count.jpg)进行保存，按ESC键退出图片采集

2.运行main.py

​	在控制台选择1，进行模型的训练，模型将保存在trainModel文件夹

​	在控制台选择2，即对窗口的人脸进行识别然后给出相应的概率

