import cv2
import numpy as np
import math

def RGB_TO_HSI(img):

    with np.errstate(divide='ignore', invalid='ignore'):

        #Load image with 32 bit floats as variable type
        bgr = np.float32(img)/255

        #Separate color channels
        blue = bgr[:,:,0]
        green = bgr[:,:,1]
        red = bgr[:,:,2]

        #Calculate Intensity
        def calc_intensity(red, blue, green):
            return np.divide(blue + green + red, 3)

        #Calculate Saturation
        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

            return saturation

        #Calculate Hue
        def calc_hue(red, blue, green):
            hue = np.copy(red)

            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                math.sqrt((red[i][j] - green[i][j])**2 +
                                        ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])

                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

            return hue

        #Merge channels into picture and return image
        #hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green)))
        return calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green)


def Linear_stretch(band):

    band = np.float32(band)/255

    min = np.min(band)
    max = np.max(band)

    band = (band - min) * 255 / (max - min)

    return band

def histogram(image):
    hist1 = cv2.calcHist(image,[0],None,[256],[0,256])
    hist1.show()

def image_tiling(image):
    numrows, numcols = 128, 128
    height = int(image.shape[0] / numrows)
    width = int(image.shape[1] / numcols)

    for row in range(numrows):
        for col in range(numcols):
            y0 = row * height
            y1 = y0 + height
            x0 = col * width
            x1 = x0 + width
            cv2.imwrite('tiling/%d_%d.jpg' % (row, col), image[y0:y1, x0:x1])


def hist_tiling(image):
    numrows, numcols = 128, 128
    height = int(image.shape[0] / numrows)
    width = int(image.shape[1] / numcols)

    for row in range(numrows):
        for col in range(numcols):
            y0 = row * height
            y1 = y0 + height
            x0 = col * width
            x1 = x0 + width
            cv2.imwrite('hist_tiling/%d_%d.jpg' % (row, col), image[y0:y1, x0:x1])



def histogram_equalization(image):

    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    cdf = hist.cumsum()

    # cdf의 값이 0인 경우는 mask처리를 하여 계산에서 제외
    # mask처리가 되면 Numpy 계산에서 제외가 됨
    # 아래는 cdf array에서 값이 0인 부분을 mask처리함
    cdf_m = np.ma.masked_equal(cdf, 0)

    # History Equalization 공식
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    # Mask처리를 했던 부분을 다시 0으로 변환
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    img2 = cdf[image]

    return img2