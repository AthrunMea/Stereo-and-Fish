import cv2
import numpy as np

img = cv2.imread('test.jpg',0)
print(img.shape[0],img.shape[1])