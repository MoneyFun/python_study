# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import face_recognition
from GenderDetectionTrain.face_train_use_keras import Model

def CatchUsbVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)
    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    model = Model()
    model.load_model(file_path='./model/gender.model.h5')

    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)

        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        if len(face_locations) > 0:  # 大于0则检测到人脸
            for (top, right, bottom, left) in face_locations:
                image = frame[top - 10: bottom + 10, left - 10: right + 10]
                faceID = model.face_predict(image)

                if faceID == 0:
                    cv2.rectangle(frame, (left - 10, top - 10), (right + 10, bottom + 10),
                                  color, thickness=2)

                    # 文字提示是
                    cv2.putText(frame, 'woman',
                                (left + 30, top + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽
                if faceID == 1:
                    cv2.rectangle(frame, (left - 10, top - 10), (right + 10, bottom + 10),
                                  color, thickness=2)

                    # 文字提示是
                    cv2.putText(frame, 'man',
                                (left + 30, top + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽
        # 显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break
            # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    CatchUsbVideo("detect", 0)
