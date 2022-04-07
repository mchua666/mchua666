# 处理
import cv2.cv2 as cv2
import numpy as np

image1=cv2.imread("E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg",cv2.IMREAD_UNCHANGED)
image1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
#二值化  r2返回阈值  image2二值图
r2,image2=cv2.threshold(image1,127,255,cv2.THRESH_BINARY)
#反二值化
r3,image3=cv2.threshold(image1,127,255,cv2.THRESH_BINARY_INV)
#低于threshold则为0
r4,image4=cv2.threshold(image1,127,255,cv2.THRESH_TOZERO)
#高于threshold则为0
r5,image5=cv2.threshold(image1,127,255,cv2.THRESH_TOZERO_INV)
#高于threshold则为threshold
r6,image6=cv2.threshold(image1,127,255,cv2.THRESH_TRUNC)


#平滑滤波
#均值滤波  sum(square)/25
image7=cv2.blur(image1,(5,5))
#盒子滤波  均值滤波  normalize=0 区域内像素求和
image8=cv2.boxFilter(image1,-1,(2,2))
#高斯滤波  0  方差    sigmaX=sigmaxY=0.3((ksize-1)*0.5-1)+0.8
image9=cv2.GaussianBlur(image1,(3,3),0)
#中值滤波
image10=cv2.medianBlur(image1,3)



cv2.imshow("1",image1)
cv2.imshow("2",image2)
cv2.imshow("3",image3)
cv2.imshow("4",image4)
cv2.imshow("5",image5)
cv2.imshow("6",image6)
cv2.imshow("7",image7)
cv2.imshow("8",image8)
cv2.imshow("9",image9)
cv2.imshow("10",image10)

cv2.waitKey(0)
cv2.destroyAllWindows()
