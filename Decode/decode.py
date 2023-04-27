import cv2
import numpy as np
import math
import glob

def compare(a, b):
    if (a==b).all():
        return True
    else:
        return False

def decode_1(judge):
    position1 = -1
    position2 = -1
    if compare(judge,[0,0,0,0,0]):
        position1 = 0
        position2 = 0
    elif compare(judge,[0,0,0,0,1]):
        position1 = 0
        position2 = 1
    elif compare(judge,[0,0,0,1,1]):
        position1 = 1
        position2 = 1
    elif compare(judge,[0,0,0,1,0]):
        position1 = 1
        position2 = 2
    elif compare(judge,[0,0,1,1,0]):
        position1 = 2
        position2 = 2
    elif compare(judge,[0,0,1,1,1]):
        position1 = 2
        position2 = 3
    elif compare(judge,[0,0,1,0,1]):
        position1 = 3
        position2 = 3
    elif compare(judge,[0,0,1,0,0]):
        position1 = 3
        position2 = 4
    elif compare(judge,[0,1,1,0,0]):
        position1 = 4
        position2 = 4
    elif compare(judge,[0,1,1,0,1]):
        position1 = 4
        position2 = 5
    elif compare(judge,[0,1,1,1,1]):
        position1 = 5
        position2 = 5
    elif compare(judge,[0,1,1,1,0]):
        position1 = 5
        position2 = 6
    elif compare(judge,[0,1,0,1,0]):
        position1 = 6
        position2 = 6
    elif compare(judge,[0,1,0,1,1]):
        position1 = 6
        position2 = 7
    elif compare(judge,[0,1,0,0,1]):
        position1 = 7
        position2 = 7
    elif compare(judge,[0,1,0,0,0]):
        position1 = 7
        position2 = 8
    elif compare(judge,[1,1,0,0,0]):
        position1 = 8
        position2 = 8
    elif compare(judge,[1,1,0,0,1]):
        position1 = 8
        position2 = 9
    elif compare(judge,[1,1,0,1,1]):
        position1 = 9
        position2 = 9
    elif compare(judge,[1,1,0,1,0]):
        position1 = 9
        position2 = 10
    elif compare(judge,[1,1,1,1,0]):
        position1 = 10
        position2 = 10
    elif compare(judge,[1,1,1,1,1]):
        position1 = 10
        position2 = 11
    elif compare(judge,[1,1,1,0,1]):
        position1 = 11
        position2 = 11
    elif compare(judge,[1,1,1,0,0]):
        position1 = 11
        position2 = 12
    elif compare(judge,[1,0,1,0,0]):
        position1 = 12
        position2 = 12
    elif compare(judge,[1,0,1,0,1]):
        position1 = 12
        position2 = 13
    elif compare(judge,[1,0,1,1,1]):
        position1 = 13
        position2 = 13
    elif compare(judge,[1,0,1,1,0]):
        position1 = 13
        position2 = 14
    elif compare(judge,[1,0,0,1,0]):
        position1 = 14
        position2 = 14
    elif compare(judge,[1,0,0,1,1]):
        position1 = 14
        position2 = 15
    elif compare(judge,[1,0,0,0,1]):
        position1 = 15
        position2 = 15
    elif compare(judge,[1,0,0,0,0]):
        position1 = 15
        position2 = 16
    return position1,position2

def binary(image,threshold_array):
    width = image.shape()[1]
    height = image.shape()[0]
    image_new = np.empty([height,width])
    for i in range(height):
        for j in range(width):
            if image[i,j] <= threshold_array[i,j]:
                image_new[i,j] = 0
            elif image[i,j] > threshold_array[i,j]:
                image_new[i,j] = 255
    return image_new

def undestort(left_camera_matrix, left_distortion,right_camera_matrix, right_distortion, size, R,T,imgL,imgR):
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                      right_camera_matrix, right_distortion, size, R,
                                                                      T)
    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size,
                                                         cv2.CV_16SC2)
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    return img1_rectified,img2_rectified

