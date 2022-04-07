# 处理
import cv2.cv2 as cv2
import numpy as np

image1=cv2.imread("E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg",cv2.IMREAD_UNCHANGED)
image1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)


#sobel 梯度边缘提取  卷积核竖向[[-1,-2,-1],[0,0,0],[1,2,1]]
sobelx=cv2.Sobel(image1,cv2.CV_64F,1,0,ksize=3)
sobely=cv2.Sobel(image1,cv2.CV_64F,0,1,ksize=3)
sobelx=cv2.convertScaleAbs(sobelx) #负值取正
sobely=cv2.convertScaleAbs(sobely)

sobelxy=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)

#scharr 梯度边缘提取  卷积核竖向[[-3,-10,-3],[0,0,0],[3,10,3]]
scharrx=cv2.Sobel(image1,cv2.CV_64F,1,0)
scharry=cv2.Sobel(image1,cv2.CV_64F,0,1)
scharrx=cv2.convertScaleAbs(scharrx) #负值取正
scharry=cv2.convertScaleAbs(scharry)

scharrxy=cv2.addWeighted(scharrx,0.5,scharry,0.5,0)


#拉普拉斯 梯度边缘提取 1 拉普拉斯图像梯度[[0,1,0],[1,-4,1],[0,1,0]]
image2=cv2.Laplacian(image1,cv2.CV_64F)
image2=cv2.convertScaleAbs(image2)

#拉普拉斯 梯度边缘提取 2 拉普拉斯图像梯度[[0,1,0],[1,-4,1],[0,1,0]]
f=np.array([[0,1,0],[1,-4,1],[0,1,0]])
image3=cv2.filter2D(image1,-1,f)



#Canny边缘检测
#sobel梯度大小 0.5|x|+0.5|y|
#高斯滤波   梯度方向 arctan(Y/x) 同方向上保留最大梯度
#去噪--->梯度---->非极大抑制------->滞后阈值----->out

image4=cv2.Canny(image1,100,200)


cv2.imshow("1",image1)
cv2.imshow("2",sobelxy)
cv2.imshow("3",scharrxy)
cv2.imshow("4",image2)
cv2.imshow("5",image3)
cv2.imshow("6",image4)

cv2.waitKey(0)
cv2.destroyAllWindows()
