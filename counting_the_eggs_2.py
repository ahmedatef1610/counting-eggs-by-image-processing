import cv2 as cv
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt

######################################################
image = cv.imread('./images/stage1_egg_3.png')
cv.imshow('input', image)
cv.waitKey()

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
cv.waitKey()

blured = cv.GaussianBlur(gray, (3,3), None)
cv.imshow('blured', blured)
cv.waitKey()

threshold, binary = cv.threshold(blured, 200, 255, cv.THRESH_BINARY)
cv.imshow('binary', binary)
cv.waitKey()

contours, hierarchies = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# print(len(contours)) # 220
contour_image = image.copy()
cv.drawContours(contour_image, contours, -1, (0,255,255), -1)
cv.imshow('contour_image_1', contour_image)
cv.waitKey()

selected_contours = []
for index, cnt in enumerate(contours):
    area = cv.contourArea(cnt)
    # print(f"index: {index} has area: {area}")
    if area > 500: 
        selected_contours.append(cnt)
    
    pass

print(f"selected_contours: {len(selected_contours)}")
print()
contour_image = image.copy()
cv.drawContours(contour_image, selected_contours, -1, (0,255,255), -1)
cv.imshow('selected_contours', contour_image)
cv.waitKey()




circle_contours = []
circle_count = 0
for index, cnt in enumerate(selected_contours):
    circle_check_image = image.copy()
    (x,y),radius = cv.minEnclosingCircle(cnt)
    x = int(x)
    y = int(y)
    circle_area = np.pi * radius**2
    contour_area = cv.contourArea(cnt)
    radius = int(radius)
    off_percentage = int(((circle_area-contour_area)/contour_area)*100)
    
    if(off_percentage) <= 50:
        is_circle = True
        circle_contours.append(cnt)
        circle_count += 1
    else:
        is_circle = False
    
    print("looking at contour index: ",index , "circle attributes: ",x,y,radius)
    print("circle area: ",circle_area,"contour area: ",contour_area,"off_percentage: ",off_percentage)
    print("stats", index , is_circle, circle_count)
    print("="*20)
     
    cv.drawContours(circle_check_image, [cnt], -1, (0,255,255), -1)
    cv.circle(circle_check_image,(x,y),radius,(255,255,255),2)
    cv.imshow('circle_check_image', circle_check_image)
    cv.waitKey()
    pass

image_output = image.copy()
cv.drawContours(image_output, circle_contours, -1, (0,255,255), 2)
cv.putText(image_output, f"number of eggs {circle_count}" , (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv.imshow('image_output', image_output)
cv.waitKey()



#####################################################################################################
cv.imwrite('./images/stage2_egg_3.png',image_output)
#####################################################################################################