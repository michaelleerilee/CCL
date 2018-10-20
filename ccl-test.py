#!/usr/bin/env python

# https://stackoverflow.com/questions/43547540/cv2-connectedcomponents-not-detecting-components

import cv2
import numpy as np

img = cv2.imread("75oB8.png",cv2.IMREAD_GRAYSCALE)
# ret, thresh = cv2.threshold(img, 127, 255, 0)
# ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# img0 = np.uint8(img)
# img0 = cv2.bitwise_not(thresh)
img0 = thresh

ret, markers = cv2.connectedComponents(img0)
# markers = markers + 1
# markers[img0 == 255] = 0
print np.amax(markers)

# markers=markers+1
# markers = cv2.watershed(img0,markers)
# img0[markers == -1] = [255,0,0]

img1 = markers.astype(np.float)/np.amax(markers)

cv2.imshow('image  ',img0); cv2.waitKey(0); cv2.destroyAllWindows()
cv2.imshow('markers',markers); cv2.waitKey(0); cv2.destroyAllWindows()
cv2.imshow('img1   ',img1); cv2.waitKey(0); cv2.destroyAllWindows()
img2=cv2.merge([img1,1.0-img1,0.5*img1])
cv2.imshow('img2   ',img2); cv2.waitKey(0); cv2.destroyAllWindows()


# https://stackoverflow.com/questions/15072736/extracting-a-region-from-an-image-using-slicing-in-python-opencv/15074748#15074748
# img2 = img[:,:,::-1]
