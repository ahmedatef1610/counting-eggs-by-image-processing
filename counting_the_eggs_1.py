import cv2 as cv
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt

######################################################

image = cv.imread('./images/egg_1.png')
cv.imshow('input', image)
cv.waitKey()

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
cv.waitKey()

blured = cv.GaussianBlur(gray, (9,9), None)
cv.imshow('blured', blured)
cv.waitKey()

threshold, binary = cv.threshold(blured, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print(threshold) # 125.0
cv.imshow('binary', binary)
cv.waitKey()

# M = np.ones((3,3))
# eroded = cv.erode(binary, M, iterations=3)
# cv.imshow('Eroded', eroded)
# cv.waitKey()
eroded = cv.erode(binary, None, iterations=1)
cv.imshow('Eroded', eroded)
cv.waitKey()
# dilated = cv.dilate(eroded, M, iterations=3)
# cv.imshow('Dilated', dilated)
# cv.waitKey()

contours, hierarchies = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# print(len(contours)) # 30
the_biggest_contour_by_area = max(contours , key=cv.contourArea)
box_contour = the_biggest_contour_by_area
# biggest_area = cv.contourArea(the_biggest_contour_by_area)

contour_image = image.copy()
cv.drawContours(contour_image, [box_contour], -1, (0,255,255), 2)
cv.imshow('contour_image_1', contour_image)
cv.waitKey()


perimeter = cv.arcLength(box_contour, True)
# print("perimeter : ", perimeter)
# print("epsilon : ", perimeter*0)
approximate_contour = cv.approxPolyDP(box_contour, perimeter*0.01, True)
contour_image_2 = image.copy()
cv.drawContours(contour_image_2, [approximate_contour], -1, (0,255,255), 2)
cv.imshow('contour_image_2', contour_image_2)
cv.waitKey()


approximate_contour_2 = cv.convexHull(box_contour)
contour_image_3 = image.copy()
cv.drawContours(contour_image_3, [approximate_contour_2], -1, (0,255,255), 2)
cv.imshow('contour_image_3', contour_image_3)
cv.waitKey()


rectangle_image = image.copy()
x,y,w,h = cv.boundingRect(box_contour)
my_hand_made_contour = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
cv.rectangle(rectangle_image, (x, y), (x+w, y+h), (0,255,255), 2)
cv.imshow('contour_image_4' , rectangle_image)
cv.waitKey()
#####################################################################################################
# mask = np.zeros(image.shape[:2], dtype='uint8')
# # cv.rectangle(mask, (x, y), (x+w, y+h), (255,255,255), -1)
# my_hand_made_contour = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
# cv.drawContours(mask, [my_hand_made_contour], -1, (255,255,255), -1)
# cv.imshow('Mask', mask)
# cv.waitKey()
#################################################
mask = np.zeros(image.shape[:2], dtype='uint8')
cv.drawContours(mask, [approximate_contour_2], -1, (255,255,255), -1)
cv.imshow('Mask', mask)
cv.waitKey()


box = cv.bitwise_and(image.copy(),image.copy(),mask=mask)
cv.imshow('box', box)
cv.waitKey()

#####################################################################################################
cv.imwrite('./images/stage1_egg_1.png',box)
#####################################################################################################





















######################################################

# import cv2 as cv
# help(cv.arcLength)
# help(cv.approxPolyDP)
# help(cv.convexHull)