import cv2
import numpy as np
import os
from cv2 import face

train = False
choose = 3


def detectFace(img):
    """
    用内置检测器检测人脸,这里选择的是haarcascade_frontalface_alt.xml
    相应的还有haarcascade_frontalcatface.xml,lbpcascade_frontalface.xml,速度和精度各有不同
    :param img:
    :return: 人脸,以及矩形框数据
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("opencvFiles/haarcascade_frontalface_alt.xml")
    faceInfo = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faceInfo) == 0:
        return None, None
    (x, y, w, h) = faceInfo[0]
    return gray[y:y + h, x:x + w], faceInfo[0]


def drawRectangle(image, rect):
    """
    画矩形框
    :param image:
    :param rect:
    :return:
    """
    (x, y, w, h) = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


def drawText(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


def prepareTrainingData(dataFoderPath):
    """
    图派你预处理,读取图片
    :param dataFoderPath:
    :return:
    """
    dirs = os.listdir(dataFoderPath)
    faces = []
    labels = []
    for dirName in dirs:
        label = int((dirName.split("_"))[0])
        subjectDirPath = dataFoderPath + "/" + dirName
        subjectImagesNames = os.listdir(subjectDirPath)
        for imageName in subjectImagesNames:
            if imageName.startswith("."):
                continue
            imagePath = subjectDirPath + "/" + imageName
            image = cv2.imread(imagePath)
            face, rect = detectFace(image)
           # cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
           #cv2.waitKey(100)
            if face is not None:
                face = cv2.resize(face, (100, 100))
                faces.append(face)
                labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels



def predict(testImg, faceRecoginzer,names):
    """
    图片预测
    :param testImg: 测试图片
    :param faceRecoginzer: opencv内置检测器
    :param names: 名字
    :return: 标签,置信度
    """
    img = testImg.copy()
    face, rect = detectFace(img)
    if face is None:
        return None, None, img
    face = cv2.resize(face, (100, 100))
    label, confidence = faceRecoginzer.predict(face)
    if confidence < 100:
        confidence = "{0}%".format(round(100 - confidence))
        drawText(img, names.get(str(label)) + "   " + confidence, rect[0], rect[1]-5)
    else:
        label = "unknown"
        confidence = "{0}%".format(round(100 - confidence))
        drawText(img, label + "   " + confidence, rect[0]+5, rect[1] -5)
    drawRectangle(img, rect)
    return label, confidence, img


if __name__ == '__main__':
    print("Preparing data...")
    faces, labels = prepareTrainingData("trainData")
    print("Data prepared...")
    print(len(faces))
    print(len(labels))
    faceRecoginzer = (cv2.face.LBPHFaceRecognizer_create() if (choose == 1) else (
        cv2.face.FisherFaceRecognizer_create() if (choose == 2) else cv2.face.EigenFaceRecognizer_create()))
    if train:
        faceRecoginzer.train(faces, np.array(labels))
        faceRecoginzer.write(r"trainModel/train.xml")
    else:
        faceRecoginzer.read("trainModel/train.xml")
        print("predictimg images...")
        testImage1 = cv2.imread("testData/test1.jpg")
        testImage2 = cv2.imread("testData/test2.jpg")

        label, confidence, predictImage1 = predict(testImage1, faceRecoginzer)
        label2, confidence2, predictImage2 = predict(testImage2, faceRecoginzer)

        cv2.imshow(str(label), cv2.resize(predictImage1, (400, 500)))
        cv2.imshow(str(label2), cv2.resize(predictImage1, (400, 500)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
