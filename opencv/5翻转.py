# 翻转

import cv2.cv2 as cv2
import numpy as np

image1=cv2.imread("E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg",cv2.IMREAD_UNCHANGED)

cols=image1.shape[0]
print("宽：",cols)
rows=image1.shape[1]
print("高：",rows)
#上下翻转
image2=cv2.flip(image1,0)
#左右翻转
image3=cv2.flip(image1,1)
#上下左右翻转
image4=cv2.flip(image1,-1)

#移动 放射变换 ->(100,200)
wart_size=np.float32([[1,0,100],[0,1,200]])
image5=cv2.warpAffine(image1,wart_size,(cols,rows))

#旋转 45度 缩放0.6
wart_size2=cv2.getRotationMatrix2D((rows/2,cols/2),45,0.6)
image6=cv2.warpAffine(image1,wart_size2,(rows,cols))

#图片菱形转换
p1=np.float32([[0,0],[cols-1,0],[0,rows-1]])
p2=np.float32([[0,rows*0.33],[cols*0.85,rows*0.25],[cols*0.15,rows*0.7]])
m=cv2.getAffineTransform(p1,p2)
image7=cv2.warpAffine(image1,m,(cols,rows))




cv2.imshow("1",image1)
cv2.imshow("2",image2)
cv2.imshow("3",image3)
cv2.imshow("4",image4)
cv2.imshow("5",image5)
cv2.imshow("6",image6)
cv2.imshow("7",image7)

cv2.waitKey(0)
cv2.destroyAllWindows()
