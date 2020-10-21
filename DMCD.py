from PIL import Image
import cv2
import converter
import numpy as np
import matplotlib.pyplot as plt


Ts = -10
imagePath = "220.jpeg"
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

Th = np.mean(mhsi) + np.std(mhsi)

for k in range(0,mhsi.shape[0]):
    for l in range(0,mhsi.shape[1]):

        if mhsi[k][l] > Th:
            sccp1[k][l] = 1
        else:
            sccp1[k][l] = 0


mrgb = np.copy(m)
sccp2 = np.copy(mrgb)

for k in range(0,mrgb.shape[0]):
    for l in range(0,mrgb.shape[1]):
        if (blue[k][l] - red[k][l]) > Ts:
            mrgb[k][l] = blue[k][l]

        else:
            mrgb[k][l] = 0

Tb = np.mean(blue) + np.std(blue)

for k in range(0,blue.shape[0]):
    for l in range(0,blue.shape[1]):

        if blue[k][l] > Tb:
            sccp2[k][l] = 1
        else:
            sccp2[k][l] = 0


#dilated_sccp1 = cv2.dilate(sccp2, kernel, iterations = 1)

erode_sccp1 = cv2.erode(sccp2, kernel, iterations = 2)
cv2.imwrite('result_mrgb.png', mrgb*255)
mrgb=np.where(mrgb>0.7,1,0)
r=np.where(erode_sccp1+mrgb==2,1,0)

sfine=np.copy(m)

Tc=0.6

for k in range(0,sfine.shape[0]):
    for l in range(0,sfine.shape[1]):
        if r[k][l] >Tc:
            sfine[k][l]=1
        else:
            sfine[k][l]=0

cv2.imwrite('result_sfine.png', sfine*255)
cv2.imwrite('result_erode.png', erode_sccp1*255)


dilate_fine = cv2.dilate(sfine, kernel, iterations = 1)
new_fine = np.copy(dilate_fine)

for k in range(0, dilate_fine.shape[0]):
    for l in range(0, dilate_fine.shape[1]):
        if dilate_fine[k][l]==1 and sfine[k][l]==0:
            new_fine[k][l] = dilate_fine[k][l]
        else:
            new_fine[k][l] = sfine[k][l]

LB=converter.Linear_stretch(blue)

stcp=np.copy(LB)
# x=np.arange(255)
# y=blue.reshape(1,)

# blue_hist= cv2.calcHist(blue, [0], None, [256], [0,256])
# plt.plot(blue_hist)
#
# plt.show()

Tbs=185

for k in range(0,stcp.shape[0]):
    for l in range(0,stcp.shape[1]):
        if LB[k][l]>Tbs:
            stcp[k][l]=1
        else:
            stcp[k][l]=0

cv2.imwrite('result_stcp.png', stcp*255)

a=new_fine+stcp
concat=np.where(a==2,1,0)

cv2.imwrite('result.png',concat*255)