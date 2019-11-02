#Importing Pkgs
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np

# Importing Image 
img = cv2.imread('images/a.png')
#plt.imshow(img)
#plt.show()

# converting to gray 
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#plt.imshow(gray)
#plt.show()

# Reducing Noise 
blur = cv2.GaussianBlur(gray, (5,5) , 0)
#plt.imshow(blur)
#plt.show()

# Edge Detection (Canny)
cannyImage = cv2.Canny(blur, 50, 150)
plt.imshow(cannyImage)
plt.show()

# Region of Interest
def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([[(200, height),(550, 250),(1100, height),]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Hough Transform
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

cropped_Image = region_of_interest(cannyImage)
rho = 1
theta = np.pi/180
threshold = 20
lines = cv2.HoughLinesP(cropped_Image,rho, theta, threshold, np.array ([]), minLineLength=20, maxLineGap=1)
line_image = display_lines(img, lines)
plt.imshow(line_image)
plt.show()

combo_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)
plt.imshow(combo_image)
plt.show()