'''
OpenCV-图片的仿射变换
把原图片映射到目标图片三个新的位置
src->dst 3 (左上角，左下角，右上角)
'''
import cv2.cv2 as cv2
import numpy as np
img = cv2.imread('E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg',1)
cv2.imshow('src',img)
#原图片的高和宽
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
#原始图片上的三个点
matSrc = np.float32([[0,0],[0,height-1],[width-1,0]])
#目标图片上的三个点
matDst = np.float32([[50,50],[300,height-200],[width-300,100]])
#组合 设置仿射变换矩阵,得到的矩阵组合，第一个参数描述的是原始图片左上角，左下角，右上角矩阵，第二个参数描述的是原来图像的三个点在目标图像中的三个位置
matAffine = cv2.getAffineTransform(matSrc,matDst)
dst = cv2.warpAffine(img,matAffine,(width,height))
cv2.imshow('dst',dst)
cv2.waitKey(0)


img = cv2.imread('E:\\LearnData\\Data1\\sx\\use_IMG\\opencv\\123.jpg')
rows, cols, ch = img.shape

pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('image', dst)
k = cv2.waitKey(0)
if k == ord('s'):
    cv2.imwrite('Rachel1.jpg', dst)
    cv2.destroyAllWindows()