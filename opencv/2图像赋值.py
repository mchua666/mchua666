import numpy as np
import cv2.cv2 as cv2
img=cv2.imread("E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg")
#灰度图赋值
img[100,100]=255.
print("灰度图赋值:",img[100,100])
#彩色
img[200,200,0]=255
print("彩色单通道赋值:",img[200,200][0])
img[101,101]=[255,0,255]
print("彩色多通道赋值:",img[101,101,1])

#获得（100,100）点2通道的值
print("获得（100,100）点2通道的值:",img.item(100,100,2))
#设置（200,200）点2通道的值
img.itemset((200,200,2),188)
print("设置（200,200）点2通道的值:",img.item((200,200,2)))

#获取图像属性
h,w,d=img.shape
print("h:",h,"w:",w,"d",d)

#获取图片大小 w*h / h*w*d
img_size=img.size
print("img_size:",img_size)

#获取图片数据类型
print("type:",img.dtype)

#感兴趣区域ROL(region of interest)
##获取面部图像
face=img[220:400,250:350]

cv2.imshow("face",face)

#通道分解合并
b=img[:,:,0]
g=img[:,:,1]
r=img[:,:,2]
#通道分解2
b2,g2,r2=cv2.split(img)

#通道合并
rgb=cv2.merge([r,g,b])

#只显示蓝色
b3=cv2.split(img)[0]
print("b3",b3)



cv2.imshow("1",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
