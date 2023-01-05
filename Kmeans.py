# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:32:49 2021

@author: User
"""


import os
import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt
import random


def openImg(fileName):
    img = Image.open(fileName)
    
    return img

def RGBtoGray(img):
    img = np.array(img)
    H, W, C = img.shape
    grayImg = np.zeros((H, W))
    
    for i in range(H):
        for j in range(W):
            grayImg[i][j] = int((img[i][j][0] + img[i][j][1]* 2 + img[i][j][2]) >> 2)
            
    grayImg = Image.fromarray(grayImg)
    
    return grayImg, H, W

# 找到像素值的最小差距的類別
def minDistance(val, x1, x2, x3):
    distArr = [abs(val-x1), abs(val-x2), abs(val-x3)]
    distArr = np.array(distArr)
    
    return np.argmin(distArr)
    
# K-means演算法
def kmeans(img, k):
    img = np.array(img)
    H, W = img.shape
    classArr = np.zeros((H, W), dtype=np.int)
    
    #print(img.shape)
    
    x1, x2, x3 = 0, 0, 0
    
    # 隨機產生K個像素值
    for i in range(k):
        x1 = random.randint(0, 256)
        x2 = random.randint(0, 256)
        x3 = random.randint(0, 256)
    
    for k in range(2000):
        x1_count, x2_count, x3_count = 0, 0, 0
        x1_val, x2_val, x3_val = 0, 0, 0
        
        tmp_x1, tmp_x2, tmp_x3 = x1, x2, x3
        
        for i in range(H):
            for j in range(W):
                classArr[i][j] = int(minDistance(img[i][j], x1, x2, x3))
                #print(classArr[i][j])
        
        for i in range(H):
            for j in range(W):
                if classArr[i][j] == 0:
                    x1_count += 1
                    x1_val += img[i][j]
                elif classArr[i][j] == 1:
                    x2_count += 1
                    x2_val += img[i][j]
                elif classArr[i][j] == 2:
                    x3_count += 1
                    x3_val += img[i][j]
                    
        x1 = int(x1_val/x1_count)
        x2 = int(x2_val/x2_count)
        x3 = int(x3_val/x3_count)
        
        if (x1-tmp_x1)**2 + (x2-tmp_x2)**2 + (x3-tmp_x3)**2 < 10:
            break
    
    
    return classArr



K = 3
img = openImg('example_512.png')
grayImg, H, W = RGBtoGray(img)
img = np.array(img)

classArr = kmeans(grayImg, K)
print(classArr.shape)
#print(classArr)

x1 = np.zeros((H, W, 3), dtype=np.int)
x2 = np.zeros((H, W, 3), dtype=np.int)
x3 = np.zeros((H, W, 3), dtype=np.int)

for i in range(H):
    for j in range(W):
        if classArr[i][j] == 0:
            x1[i][j][0] = img[i][j][0]
            x1[i][j][1] = img[i][j][1]
            x1[i][j][2] = img[i][j][2]
        elif classArr[i][j] == 1:
            x2[i][j][0] = img[i][j][0]
            x2[i][j][1] = img[i][j][1]
            x2[i][j][2] = img[i][j][2]
        elif classArr[i][j] == 2:
            x3[i][j][0] = img[i][j][0]
            x3[i][j][1] = img[i][j][1]
            x3[i][j][2] = img[i][j][2]
        

print(x1.shape, x2.shape, x3.shape)

x1Img = Image.fromarray(np.uint8(x1))
x2Img = Image.fromarray(np.uint8(x2))
x3Img = Image.fromarray(np.uint8(x3))

plt.figure()
f, axarr = plt.subplots(4,1) 
#plt.imshow(img)
axarr[0].imshow(img)
axarr[1].imshow(x1Img)
axarr[2].imshow(x2Img)
axarr[3].imshow(x3Img)

x1Img.save('x1.png')
x2Img.save('x2.png')
x3Img.save('x3.png')

plt.show()






