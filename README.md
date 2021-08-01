# counting-eggs-by-image-processing

this project is doing operation by two stage 

### first stage

it detects box of eggs in image by [contours](https://docs.opencv.org/4.5.2/d4/d73/tutorial_py_contours_begin.html) method

![stage1](https://user-images.githubusercontent.com/39852784/127775555-1c337e6c-fa55-4c27-97e2-8a2c72aa3a4a.gif)


### second stage

it counts the number of eggs in box by comparing the area of contours and minimum area of a circle it can content this contour

![stage2](https://user-images.githubusercontent.com/39852784/127775569-418d27b2-85b3-473a-8242-2c8453810a44.gif)

![info](https://user-images.githubusercontent.com/39852784/127775707-60c7033a-6d52-4321-880c-02539793b223.png)

