# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import sys

if len(sys.argv) != 3:
	print("%s src_dir dst_dir" % sys.argv[0])
	quit()

src_dir = sys.argv[1]
dst_dir = sys.argv[2]
print(src_dir)
print(dst_dir)


# 按照指定图像大小调整尺寸
def resize_image(image):
    top, bottom, left, right = (0, 0, 0, 0)

    # 获取图像尺寸
    h, w, _ = image.shape

    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left

        # RGB颜色
    BLACK = [0, 0, 0]

    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return constant


def resize_images(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            newdir = os.path.join(dst_dir, dir_item)
            print(newdir)
            os.makedirs(newdir)
            for jpg_item in os.listdir(full_path):
                if jpg_item.endswith('.jpg'):
                    img_path = os.path.join(full_path, jpg_item)
                    image = cv2.imread(img_path)
                    image = resize_image(image)
                    #cv2.imshow("show", image)
                    #cv2.waitKey(1000)

                    newimg_path = os.path.join(newdir, jpg_item)
                    cv2.imwrite(newimg_path, image)


if __name__ == '__main__':
    resize_images(src_dir)
