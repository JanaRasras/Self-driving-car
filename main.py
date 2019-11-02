# importing packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2  # conda install -c menpo opencv

# reading the image
image = cv2.imread('images\solidWhiteCurve.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.show()


# Apply Gaussian 
kernel_size = 15
gaussian_img = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
plt.imshow(gaussian_img)
plt.show()


# Canny Edge Detection
low_threshold = 50
high_threshold = 150
canny_edges = cv2.Canny(gaussian_img, low_threshold, high_threshold)
plt.imshow(canny_edges)
plt.show()

#
def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([[(200, height),(550, 250),(1100, height),]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image
cropped_Image = region_of_interest(canny_edges)
rho = 2
theta = np.pi/180
threshold = 100
lines = cv2.HoughLinesP(cropped_Image,rho, theta, threshold, np.array ([]), minLineLength=40, maxLineGap=5)
line_image = display_lines(image, lines)
plt.imshow(line_image)
plt.show()

#
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image
cropped_Image = region_of_interest(canny_edges)
rho = 2
theta = np.pi/180
threshold = 100
lines = cv2.HoughLinesP(cropped_Image,rho, theta, threshold, np.array ([]), minLineLength=40, maxLineGap=5)
line_image = display_lines(image, lines)
plt.imshow(line_image)
plt.show()

#
combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
plt.imshow(combo_image)
plt.show()