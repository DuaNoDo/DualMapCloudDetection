from PIL import Image
import cv2
import converter
import numpy as np
import matplotlib.pyplot as plt


Ts = -10

image = Image.open(imagePath)

bgr = np.float32(image)/255

# BGR 밴드 분할
blue = bgr[:,:,0]
green = bgr[:,:,1]
red = bgr[:,:,2]

# RGB 이미지를 HSI 로 변환
h, s, i = converter.RGB_TO_HSI(image)

# dilate 할 커널 생성
kernel_size_row = 3
kernel_size_col = 3
kernel = np.ones((3, 3), np.uint8)


m = np.copy(h)
mhsi = np.copy(m)


for k in range(0,m.shape[0]):
    for l in range(0,m.shape[1]):
        m[k][l]=i[k][l]/(h[k][l]+1)


for k in range(0,m.shape[0]):
    for l in range(0,m.shape[1]):

        if blue[k][l] - red[k][l] > Ts:
            mhsi[k][l] = m[k][l]

        else:
            mhsi[k][l] = 0


# 구름탐지

sccp1 = np.copy(mhsi)

Th = np.mean(mshi) + np.std(mhsi)

for k in range(0,mshi.shape[0]):
    for l in range(0,mshi.shape[1]):

        if mhsi[k][l] > Th:
            sccp1 = 1
        else:
            sccp1 = 0


mrgb = np.copy(m)

sccp2 = np.copy(mrgb)

Th = np.mean(blue) + np.std(blue)

for k in range(0,mrgb.shape[0]):
    for l in range(0,mrgb.shape[1]):
        if blue[k][l] - red[k][l] > Ts:
            mrgb = blue[k][l]

        else:
            mrgb = 0


for k in range(0,blue.shape[0]):
    for l in range(0,blue.shape[1]):

        if blue[k][l] > Tb:
            sccp2 = 1
        else:
            sccp2 = 0

dilated_sccp1 = cv2.dilate(sccp1, kernel, iterations = 1)

r=sccp2/dilated_sccp1

sfine=np.copy(m)

Tc=0.6

for k in range(0,sfine.shape[0]):
    for l in range(0,sfine.shape[1]):
        if r>Tc:
            sfine[k][l]=1
        else:
            sfine[k][l]=0


LB=converter.Linear_stretch(blue)

stcp=np.copy(LB)
x=np.arange(255)
y=blue.reshape(1)

Tb=

for k in range(0,stcp.shape[0]):
    for l in range(0,stcp.shape[1]):
        if LB[k][l]>Tb:
            stcp[k][l]=1
        else:
            stcp[k][l]=0