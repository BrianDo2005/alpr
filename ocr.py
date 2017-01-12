__author__ = 'Samar'

import sys
import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2
images = glob.glob('plates/plate*.jpg')

digits_train = np.load('digit-samples.npy')[:300]
labels_train = np.load('digit-labels.npy')[:300]

digits_test = np.load('digit-samples.npy')[300:]
labels_test = np.load('digit-labels.npy')[300:]

print digits_train.shape, labels_train.shape
print digits_test.shape, labels_test.shape

knn = cv2.ml.KNearest_create()
knn.train(digits_train, cv2.ml.ROW_SAMPLE, labels_train)

result = knn.findNearest(digits_test, k=1)

mask = labels_test == result[1]
correct = np.count_nonzero(mask)
print "Accuracy:", correct/float(result[1].size)

contour_list = []

for image in images:
    colour = cv2.imread(image)
    im = cv2.imread(image)[:,:,0]
    ret, thresh = cv2.threshold(im, np.mean(im), 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('Threshold', thresh)
    print

    ret,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        [x,y,w,h] = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        M = cv2.moments(cnt)
        rect_area = w*h
        aspect_ratio = float(h)/w
        if area>50 and aspect_ratio > 1:
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(20,20))
            sample = roismall.reshape((1,400))
            sample = np.array(sample,np.float32)
            pred = knn.findNearest(sample, k=1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(colour,chr(pred[1]).upper(),(x,y+h), font, 1,(0,255,0),2,cv2.LINE_AA)
            cv2.imshow('text', colour)

            print chr(pred[1]).upper(),


    cv2.waitKey(0)