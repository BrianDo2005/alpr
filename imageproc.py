__author__ = 'Samar'
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

images = glob.glob("cars/*.jpg")
image = cv2.imread(images[1])

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (3,3),0)
image = cv2.equalizeHist(image)


cv2.imshow("Image", image)

cv2.waitKey(0)