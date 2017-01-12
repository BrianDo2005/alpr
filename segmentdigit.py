__author__ = 'Samar'
import sys
import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2

images = glob.glob('plates/plate*.jpg')
print len(images)
count = 0

samples = np.empty((0,400), dtype=np.float32)
responses = []
numbers = [i for i in range(48,58)]
letters = [j for j in range(97, 123)]
alphanumeric = numbers + letters
classes = {}
for i in range(len(alphanumeric)):
    classes[i] = alphanumeric[i]


for image in images:
    count += 1
    print "\n\nPlate #", count
    colour = cv2.imread(image)
    im = cv2.imread(image)[:,:,0]
    #im = cv2.GaussianBlur(im,(3,3),0)

    #thresh = cv2.adaptiveThreshold(im,255,1,1,11,2)
    ret, thresh = cv2.threshold(im, np.mean(im), 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('Threshold', thresh)

    ret,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        M = cv2.moments(cnt)
        rect_area = w*h
        aspect_ratio = float(w)/h

        if area>50 and h>20:

            cv2.rectangle(colour,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(20,20))
            cv2.imshow('Localized', colour)
            key = cv2.waitKey(0)

            if key == 27:
               break
            elif key in alphanumeric:
                print chr(key),
                responses.append(key)
                sample = roismall.reshape((1,400))
                samples = np.append(samples,sample,0)


responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))

#np.save('digit-samples', samples)
#np.save('digit-labels', responses)

print
print "Samples type:", samples.dtype, "Responses type:", responses.dtype
print "Samples shape:", samples.shape, "Responses shape:", responses.shape

#cfor i in responses: print chr(i)
print "Training complete!"
