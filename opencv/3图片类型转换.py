# 图像类型转换
import cv2.cv2 as cv2
image1=cv2.imread("E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg",cv2.IMREAD_UNCHANGED)

#转换成灰度
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)

#BGR转换成RGB
image3=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)

#灰度图转换成BGR
image4=cv2.cvtColor(image2,cv2.COLOR_GRAY2BGR)


cv2.imshow("1",image1)
cv2.imshow("2",image2)
cv2.imshow("3",image3)
cv2.imshow("4",image4)

cv2.waitKey(0)
cv2.destroyAllWindows()
