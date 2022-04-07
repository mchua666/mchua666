# 处理
import cv2.cv2 as cv2
import numpy as np

image1=cv2.imread("E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg",cv2.IMREAD_UNCHANGED)

#缩放
image2=cv2.resize(image1,(200,100))

print("原图维度：",image1.shape)
# 按比例缩放
cols=image1.shape[0]
print("宽：",cols)
rows=image1.shape[1]
print("高：",rows)
image3=cv2.resize(image1,(round(cols*0.5),round(rows*0.5)))

#按比例缩放参数版
image4=cv2.resize(image1,None,fx=1.2,fy=0.5)

cv2.imshow("1",image1)
cv2.imshow("2",image2)
cv2.imshow("3",image3)
cv2.imshow("4",image4)

cv2.waitKey(0)
cv2.destroyAllWindows()
