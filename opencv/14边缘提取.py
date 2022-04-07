#1.高斯滤波器提取边缘特征
import cv2.cv2 as cv2
img = cv2.imread("E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg",0)
blurred = cv2.GaussianBlur(img,(11,11),0) #高斯矩阵的长与宽都是11，标准差为0
gaussImg = img - blurred
cv2.imshow("Image",gaussImg)
cv2.waitKey(0)


# 2.Canny边缘特征提取
import cv2.cv2 as cv2
img = cv2.imread("E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg",0)
blurred = cv2.GaussianBlur(img,(11,11),0)
gaussImg = cv2.Canny(blurred, 10, 70)
cv2.imshow("Img",gaussImg)
cv2.waitKey(0)


import cv2.cv2 as cv2
import numpy as np

# imread()两个参数：
# 1、图片路径。
# 2、读取图片的形式（1：默认值，加载彩色图片。 0：加载灰度图片。 -1：加载原图片）
img = cv2.imread(r"E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg")
cv2.imshow('img', img)

ret, thresh1 = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)    # 阈值分割，黑白二值
ret, thresh2 = cv2.threshold(thresh1, 80, 255, cv2.THRESH_BINARY_INV)    # （黑白二值反转）

cv2.imshow('img1', thresh1)
cv2.imshow('img2', thresh2)

# Canny算子是双阈值，所以需要指定两个阈值，阈值越小，边缘越丰富。
img3 = cv2.Canny(img, 80, 255)
# 对img3图像进行反转
img4 = cv2.bitwise_not(img3)
cv2.imshow('img4', img4)

cv2.waitKey()
# 关闭窗口并取消分配任何相关的内存使用
cv2.destroyAllWindows()
