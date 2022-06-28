import numpy as  np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import copy


matplotlib.use('TkAgg')

img = cv2.imread('../exsrc2.png')
orig_img = img.copy()
height, width, channel = img.shape
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(12,8))
plt.subplot(121),plt.imshow(img[:,:,::-1],'gray')
plt.subplot(122),plt.imshow(imgray,'gray')
plt.axis('off')
plt.savefig("Start")


# Edge를 뚜렷하게 하기 위해 가우시안 블러 적용
blur = cv2.GaussianBlur(imgray,(5,5),0)

# Adaptive Threshold 적용
thr = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
plt.figure(figsize=(20,20))

# dilation - erode with / without blur
kernel = np.ones((3,3),np.uint8)
dil = cv2.dilate(blur,kernel,iterations=1)
ero = cv2.erode(blur,kernel,iterations=1)
morph = dil - ero

kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

topHat = cv2.morphologyEx(imgray, cv2.MORPH_TOPHAT, kernel2)
blackHat = cv2.morphologyEx(imgray, cv2.MORPH_BLACKHAT, kernel2)

imgGrayscalePlusTopHat = cv2.add(imgray, topHat)
subtract = cv2.subtract(imgGrayscalePlusTopHat, blackHat)
thr2 = cv2.adaptiveThreshold(subtract,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY,11,2)
plt.figure(figsize=(12,8))
plt.subplot(221), plt.imshow(blur,'gray')
plt.title("blurred")
plt.subplot(222), plt.imshow(thr,'gray')
plt.title("after Adaptive Threshold")
plt.subplot(223), plt.imshow(morph,'gray')
plt.title("Dilation - Erode (with blur)")
plt.subplot(224), plt.imshow(thr2,'gray')

plt.title("top-black AT")
plt.savefig("Preprocess")
plt.show()

# canny 하지 않고
# 그냥 Adaptive Thresh가 젤 잘나옴
orig_img = img.copy()
cnts, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_dict = []
pos_cnt = list()
box1 = list()

contours_dict = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(orig_img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

    # insert to dict
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

plt.figure(figsize=(12, 8))
plt.imshow(orig_img[:, :, ::-1])
plt.savefig("Contour_candidates")
plt.show()