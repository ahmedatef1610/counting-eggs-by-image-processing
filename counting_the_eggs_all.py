import cv2 as cv
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt

######################################################
image = cv.imread('./images/egg_3.png')
cv.imshow('input', image)
cv.waitKey(200)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)
cv.waitKey(200)

blured = cv.GaussianBlur(gray, (9,9), None)
cv.imshow('blured', blured)
cv.waitKey(200)

threshold, binary = cv.threshold(blured, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print(threshold) # 125.0
cv.imshow('binary', binary)
cv.waitKey(200)

eroded = cv.erode(binary, None, iterations=1)
cv.imshow('Eroded', eroded)
cv.waitKey(200)


contours, hierarchies = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# print(len(contours)) # 30
the_biggest_contour_by_area = max(contours , key=cv.contourArea)
box_contour = the_biggest_contour_by_area
# biggest_area = cv.contourArea(the_biggest_contour_by_area)
contour_image = image.copy()
cv.drawContours(contour_image, [box_contour], -1, (0,255,255), 2)
cv.imshow('contour_image_1', contour_image)
cv.waitKey(200)


perimeter = cv.arcLength(box_contour,True)
approximate_contour = cv.approxPolyDP(box_contour,perimeter*0.04,True)
contour_image_2 = image.copy()
cv.drawContours(contour_image_2, [approximate_contour], -1, (0,255,255), 2)
# cv.imshow('contour_image_2', contour_image_2)
# cv.waitKey(200)

approximate_contour_2 = cv.convexHull(box_contour)
contour_image_3 = image.copy()
cv.drawContours(contour_image_3, [approximate_contour_2], -1, (0,255,255), 2)
cv.imshow('contour_image_3', contour_image_3)
cv.waitKey(200)
################################################
# img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# vertices = np.array([[250,700],[425,400],[600,700]],np.int32)
# # pts = vertices.reshape((-1,1,2))

# cv.polylines(img_rgb,[vertices],isClosed=True,color=(0,0,255),thickness=20)
# plt.imshow(img_rgb)
#############
# img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# vertices = np.array([[250,700],[425,400],[600,700]],np.int32)
# # pts = vertices.reshape((-1,1,2))

# cv.fillPoly(img_rgb,[vertices],color=(0,0,255))
# plt.imshow(img_rgb)
################################################

rectangle_image = image.copy()
x,y,w,h = cv.boundingRect(box_contour)
my_hand_made_contour = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
cv.rectangle(rectangle_image, (x, y), (x+w, y+h), (0,255,255), 2)
# cv.imshow('contour_image_4' , rectangle_image)
# cv.waitKey(200)
#####################################################################################################
# mask = np.zeros(image.shape[:2], dtype='uint8')
# # cv.rectangle(mask, (x, y), (x+w, y+h), (255,255,255), -1)
# my_hand_made_contour = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
# cv.drawContours(mask, [my_hand_made_contour], -1, (255,255,255), -1)
# cv.imshow('Mask', mask)
# cv.waitKey(200)
#################################################
mask = np.zeros(image.shape[:2], dtype='uint8')
cv.drawContours(mask, [approximate_contour_2], -1, (255,255,255), -1)
cv.imshow('Mask', mask)
cv.waitKey(200)


box = cv.bitwise_and(image.copy(),image.copy(),mask=mask)
cv.imshow('box', box)
cv.waitKey(200)

#####################################################################################################
#####################################################################################################
#####################################################################################################
image = box
cv.imshow('input 2', image)
cv.waitKey(200)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('gray 2', gray)
cv.waitKey(200)

blured = cv.GaussianBlur(gray, (3,3), None)
cv.imshow('blured 2', blured)
cv.waitKey(200)

threshold, binary = cv.threshold(blured, 200, 255, cv.THRESH_BINARY)
cv.imshow('binary 2', binary)
cv.waitKey(200)

contours, hierarchies = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# print(len(contours)) # 220
contour_image = image.copy()
cv.drawContours(contour_image, contours, -1, (0,255,255), -1)
cv.imshow('contour_image_1 2', contour_image)
cv.waitKey(200)

selected_contours = []
for index, cnt in enumerate(contours):
    area = cv.contourArea(cnt)
    # print(f"index: {index} has area: {area}")
    if area > 500: 
        selected_contours.append(cnt)
    
    pass

print(f"selected_contours: {len(selected_contours)}")
contour_image = image.copy()
cv.drawContours(contour_image, selected_contours, -1, (0,255,255), -1)
cv.imshow('selected_contours', contour_image)
cv.waitKey(200)




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
    
    cv.drawContours(circle_check_image, [cnt], -1, (0,255,255), -1)
    cv.circle(circle_check_image,(x,y),radius,(255,255,255),2)
    cv.imshow('circle_check_image', circle_check_image)
    cv.waitKey(200)
    pass

image_output = image.copy()
cv.drawContours(image_output, circle_contours, -1, (0,255,255), 2)
cv.putText(image_output, f"number of eggs {circle_count}" , (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
cv.imshow('image_output', image_output)
cv.waitKey()