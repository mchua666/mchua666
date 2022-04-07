# 处理
import cv2.cv2 as cv2
import numpy as np

image=cv2.imread("E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg",cv2.IMREAD_UNCHANGED)


image1=cv2.pyrDown(image)


image2=cv2.pyrDown(image1)

image3=cv2.pyrUp(image2)
# image3=image2-image1


cv2.imshow("1",image1)
cv2.imshow("2",image2)
cv2.imshow("3",image3)



cv2.waitKey(0)
cv2.destroyAllWindows()
