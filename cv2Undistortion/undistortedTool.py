import cv2

def undistroted(img1, img2, left_map1, left_map2, right_map1, right_map2):

    img1_rectified = cv2.remap(img1, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(img2, right_map1, right_map2, cv2.INTER_LINEAR)

    return img1_rectified, img2_rectified

def map(K1, D1, K2, D2, picture_size, Rotation, Translation):
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1, D1, K2, D2, picture_size,
                                                                      Rotation, Translation)
    left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, picture_size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, picture_size, cv2.CV_16SC2)
    return left_map1, left_map2, right_map1, right_map2

