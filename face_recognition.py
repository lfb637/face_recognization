# -*- coding: utf-8 -*-
import json
import cv2
from face_train import Model
import os


class Face_recognition():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def __init__(self):
        with open('contrast_table', 'r') as f:
            self.contrast_table = json.loads(f.read())
        self.model = Model()
        # 加载训练数据
        self.model.load_model(file_path='F:/bs_data/lfb.h5')
        # 框住人脸的矩形边框颜色
        self.color = (0, 255, 0)

        # 捕获指定摄像头的实时视频流
        self.cap = cv2.VideoCapture(0)
        # 调用人脸识别器
        self.cascade_path = "D:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml"

    def recongition(self):
        while True:
            ret, frame = self.cap.read()  # 读取一帧视频
            if ret is True:
                # 图像灰化，降低计算复杂度
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue
            # 使用人脸识别分类器，读入分类器
            cascade = cv2.CascadeClassifier(self.cascade_path)
            # 利用分类器识别出哪个区域为人脸
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect
                    # 截取脸部图像提交给模型识别这是谁
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    probability, name_number = self.model.face_predict(image)
                    print("name_number:", name_number)
                    name = self.contrast_table[str(name_number)]
                    print('name:', name)

                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), self.color, thickness=2)

                    # 标出识别者
                    # cv2.putText(frame, name, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    if probability > 0.7:
                         cv2.putText(frame, name, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    else:
                         cv2.putText(frame, 'unknow', (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            cv2.imshow("face_recognition", frame)

            # 等待10毫秒看是否有按键输入
            k = cv2.waitKey(10)
            if k & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = Face_recognition()
    fr.recongition()
# Face_Recognization
# 整个项目分为三个模块
1、 人脸数据集的建立
2、 网络模型的建立
3、 训练神经网络并进行识别

技术路线：Tensorflow、Keras、Opencv
数据集：本地采集（200*3）
样本类别	        损失值（loss）	      准确率（acc）
测试集	        0.1392	              96.11%
训练集	        0.1220	              96.67%
https://github.com/lfb637/Face_Recognization.git