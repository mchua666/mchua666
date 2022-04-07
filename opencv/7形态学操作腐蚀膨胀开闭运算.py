# 图像类型转换
import cv2.cv2 as cv2
import numpy as np

image1=cv2.imread("C:\\Users\\86132\\Desktop\\123.png",cv2.IMREAD_UNCHANGED)
image1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
#图像腐蚀  k为全1卷积核
k=np.ones((5,5),np.uint8)
image2=cv2.erode(image1,k,iterations=2)

#图像膨胀
k1=np.ones((5,5),np.uint8)
image3=cv2.dilate(image1,k1,iterations=2)

#图像开运算  先腐蚀后膨胀
k2=np.ones((5,5),np.uint8)
image4=cv2.morphologyEx(image1,cv2.MORPH_OPEN,k2,iterations=2)
#图像闭运算  先膨胀后腐蚀
k3=np.ones((5,5),np.uint8)
image5=cv2.morphologyEx(image1,cv2.MORPH_CLOSE,k3,iterations=2)

#图像梯度运算  膨胀-腐蚀
k4=np.ones((5,5),np.uint8)
image6=cv2.morphologyEx(image1,cv2.MORPH_GRADIENT,k4)


#图像高帽运算  原图-开运算
k5=np.ones((5,5),np.uint8)
image7=cv2.morphologyEx(image1,cv2.MORPH_TOPHAT,k5)


#图像黑帽运算  闭运算-原图
k6=np.ones((10,10),np.uint8)
image8=cv2.morphologyEx(image1,cv2.MORPH_BLACKHAT,k6)


cv2.imshow("1",image1)
cv2.imshow("2",image2)
cv2.imshow("3",image3)
cv2.imshow("4",image4)
cv2.imshow("5",image5)
cv2.imshow("6",image6)
# cv2.imshow("3",image3)
# cv2.imshow("4",image4)

cv2.waitKey(0)
cv2.destroyAllWindows()
