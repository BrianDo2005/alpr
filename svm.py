__author__ = 'Samar'
import numpy as np
import sys
import glob
import cv2
from sklearn.utils import shuffle

samples = np.load('train_samples.npy')
labels = np.load('train_labels.npy')
test_samples = np.load('test_samples.npy')
test_labels = np.load('test_labels.npy')
test_labels = test_labels.astype(np.float32)

samples, labels = shuffle(samples, labels, random_state=42)

testplate = cv2.imread('plate-P5280057.jpg')[:,:,0]
testplate = cv2.Sobel(testplate,cv2.CV_8U,1,0,ksize=1)

nottestplate = cv2.imread('notplate-P1010008.jpg')[:,:,0]
nottestplate = cv2.Sobel(nottestplate,cv2.CV_8U,1,0,ksize=1)
#cv2.imshow("Tests:", np.hstack((testplate, nottestplate)))
#cv2.waitKey()

testplate = testplate.astype(np.float32)
nottestplate = nottestplate.astype(np.float32)

#print testplate.dtype, nottestplate.dtype

testplate = testplate.reshape(1, 40*180)
nottestplate = nottestplate.reshape(1, 40*180)
#print testplate.shape

model = cv2.ml.SVM_create()

model.setKernel(cv2.ml.SVM_LINEAR)
print model.getKernelType(), model.getType(), model.getC(), model.getGamma()
model.train(samples, cv2.ml.ROW_SAMPLE, labels)
model.save('svm_data.dat')

#for i in dir(model): print i,

print model.isClassifier()
print model.isTrained()

knn = cv2.ml.KNearest_create()
knn.train(samples, cv2.ml.ROW_SAMPLE, labels)

result = knn.findNearest(test_samples, k=7)
print result[1]

#result = model.predict(test_samples)
print result[1].dtype
print test_labels.dtype
mask = test_labels == result[1]
correct = np.count_nonzero(mask)
print "Accuracy:", correct/float(result[1].size)

result2 = knn.findNearest(np.vstack((testplate, nottestplate)), k=7)
print result