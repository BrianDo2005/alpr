__author__ = 'Samar'
import cv2
import numpy as np
import glob
import time
from skimage.transform import pyramid_gaussian

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

(winW, winH) = (180, 40)

cars = glob.glob("cars/*.jpg")
print cars

samples = np.load('train_samples.npy')
labels = np.load('train_labels.npy')
knn = cv2.ml.KNearest_create()
knn.train(samples, cv2.ml.ROW_SAMPLE, labels)

'''
colour = cv2.imread('P1010004.jpg')
image = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (3,3),0)
image = cv2.equalizeHist(image)
image =  cv2.Sobel(image,cv2.CV_8U,1,0,ksize=1)
'''
for picture in cars:
    colour = cv2.imread(picture)
    image = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3,3),0)
    image = cv2.equalizeHist(image)
    image =  cv2.Sobel(image,cv2.CV_8U,1,0,ksize=1)
    for (x, y, window) in sliding_window(image, stepSize=20, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if cv2.waitKey(1) == 27:
            break
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        result = knn.findNearest(window.reshape(1, 40*180).astype(np.float32), k=6)
        if result[1][0][0] == 1:
            print "PLATE"
            time.sleep(1)

        clone = colour.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)