__author__ = 'Samar'
import sys
import glob
import numpy as np
from matplotlib import pyplot as plt
import cv2

images = glob.glob('cars/*.jpg')

print len(images)

count = 0

for image in images:
    colour = cv2.imread(image)
    gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(eq,(3,3),0)
    lap = cv2.Laplacian(blur, cv2.CV_8U)

    #cv2.imshow('Laplacian', lap)
    #cv2.waitKey(0)
    ret,contours,hierarchy = cv2.findContours(lap,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        M = cv2.moments(cnt)
        rect_area = w*h
        aspect_ratio = float(w)/h

        if area>5000 and abs(aspect_ratio - 4.5) < 1.0:
            solidity = float(area)/hull_area
            extent = float(area)/rect_area
            if solidity > 0.8:
                roi = eq[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(220,50))
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                outline = cv2.drawContours(colour,[box],0,(0,255,0),2)

                #cv2.rectangle(colour,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.imshow('Localized', colour)

                cv2.imshow('Resized', roismall)

                key = cv2.waitKey(0)
                if key == 32:
                    cv2.imwrite('plate-'+image[5:], roismall)
                    count += 1
                    print "Plate #", count
                if key == 27:
                    sys.exit()

print count