from PIL import Image
import cv2
import converter
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths

Tc = 0.6
TPc = 0.95
imagePath = "asdfasdfasd.png"


image = cv2.imread(imagePath)
gray_image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)


#cv2.imshow("canny9",cv2.Canny(gray_image, 150,200))
#cv2.waitKey(0)
converter.image_tiling(image)

imagePaths = sorted(list(paths.list_images('./tiling/')))
tiling_thumb, hist_tiling_thumb = np.ones((128, 128)), np.ones((128, 128))


for imagePath in imagePaths:

    tiling_image = cv2.imread(imagePath)
    gray_image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    canny = cv2.Canny(gray_image, 150, 200)
    bgr = np.float32(tiling_image) / 255

    # BGR 밴드 분할
    blue = bgr[:, :, 0]
    green = bgr[:, :, 1]
    red = bgr[:, :, 2]
    Tc_sum = 0
    canny_sum=0
    for k in range(0, tiling_image.shape[0]):
        for l in range(0, tiling_image.shape[1]):
            if min(red[k][l], green[k][l], blue[k][l])/max(red[k][l], green[k][l], blue[k][l]) > Tc:
                Tc_sum +=1
            if canny[k][l]!=0:
                canny_sum+=1

    imagePath = imagePath.split("/")[2]
    imagePath = imagePath.split(".")[0]
    file_name = imagePath.split("_")
    row = int(file_name[0])
    col = int(file_name[1])

    if canny_sum / (tiling_image.shape[0] * tiling_image.shape[1]) > 0.03:
        tiling_thumb[row][col]=0

    if Tc_sum / (tiling_image.shape[0]*tiling_image.shape[1]) < 0.4:
        tiling_thumb[row][col]=0


print("Tiling Process End")

hist_image = converter.histogram_equalization(image)
cv2.imwrite('hist.png', hist_image)
converter.hist_tiling(hist_image)

histPaths = sorted(list(paths.list_images('./hist_tiling/')))

for imagePath in histPaths:
    tiling_image = cv2.imread(imagePath)
    canny = cv2.Canny(cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE), 150, 200)
    minLineLength = 100
    maxLineGap = 10


    bgr = np.float32(tiling_image) / 255

    # BGR 밴드 분할
    blue = bgr[:, :, 0]
    green = bgr[:, :, 1]
    red = bgr[:, :, 2]
    TPc_sum = 0
    for k in range(0, tiling_image.shape[0]):
        for l in range(0, tiling_image.shape[1]):
            if min(red[k][l], green[k][l], blue[k][l])/max(red[k][l], green[k][l], blue[k][l]) > TPc:
                TPc_sum +=1


    imagePath = imagePath.split("/")[2]
    imagePath = imagePath.split(".")[0]
    file_name = imagePath.split("_")
    row = int(file_name[0])
    col = int(file_name[1])
    print(TPc_sum)
    if TPc_sum / (tiling_image.shape[0]*tiling_image.shape[1]) <0.4:
        hist_tiling_thumb[row][col]=0


    #cv2.imshow("show",tiling_image)

cv2.imwrite('thumb.png', tiling_thumb*255)

cv2.imwrite('hist_thumb.png', hist_tiling_thumb*255)

cv2.imwrite('final.png',np.where(tiling_thumb+hist_tiling_thumb==2 ,255,0))