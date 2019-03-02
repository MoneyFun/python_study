# -*- coding: utf-8 -*-

import cv2
from GenderDetectionTrain.face_train_use_keras import Model
import face_recognition

if __name__ == '__main__':
    # 加载模型
    model = Model()
    model.load_model(file_path='./model/gender.model.h5')

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)

    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture(0)

    # 循环检测识别人脸
    while True:
        _, frame = cap.read()  # 读取一帧视频
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) > 0:
            for faceRect in face_locations:
                top, right, bottom, left = faceRect

                # 截取脸部图像提交给模型识别
                image = frame[top - 10: bottom + 10, left - 10: right + 10]
                #cv2.imshow("face", image)
                #cv2.waitKey(1)
                #continue
                faceID = model.face_predict(image)

                # 如果是0,1
                if faceID == 0:
                    cv2.rectangle(frame, (left - 10, top - 10), (right + 10, bottom + 10),
                                  color, thickness=2)

                    # 文字提示是
                    cv2.putText(frame, 'woman',
                                (top + 30, left + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽
                if faceID == 1:
                    cv2.rectangle(frame, (left - 10, top - 10), (right + 10, bottom + 10),
                                  color, thickness=2)

                    # 文字提示是
                    cv2.putText(frame, 'man',
                                (top + 30, left + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽

        cv2.imshow("Face", frame)

        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(1)
        # 如果输入q则退出循环
        if k == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
