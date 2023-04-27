import numpy as np
import cv2
import glob

def camera_calibration(dir,chessboard_size,square_size):
    calibration_dir = dir
    calibration_files = calibration_dir+'*.png'
    image_file = glob.glob(calibration_files)
    width = chessboard_size[0]
    height = chessboard_size[1]
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp *= square_size

    imgpoints = []
    objpoints = []
    for fname in image_file:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret == True :
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            #img_corner_identify = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            imgpoints.append(corners2)
            objpoints.append(objp)
            print(fname)

    ret, K1, D1, rvecs, tvecs, = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, K1, D1, rvecs, tvecs, imgpoints, objpoints

def stereo_camera_calibration(obj, imgp_L, imgp_R, K1, D1, K2, D2, picture_size):
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = 0
    #flags |= cv2.CALIB_FIX_INTRINSIC
    retS, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(obj, imgp_L, imgp_R, K1, D1, K2, D2, picture_size, criteria_stereo, flags)
    return retS, K1, K2, D2, R, T, E, F

if __name__ == '__main__':
    ret_left, intrinsic_matrix_left, distortion_matrix_left, rotation_left, translation_left, imgp_L, obj_L = camera_calibration('left_pic/', (11, 8), 25)
    ret_right, intrinsic_matrix_right, distortion_matrix_right, rotation_right, translation_right, imgp_R, obj_R = camera_calibration('right_pic/', (11, 8), 25) #标定图片的选择影响很大，慎重选择 用img_corner_identify角点识别绘图筛选一下
    print('left:', ret_left, '\n', intrinsic_matrix_left, '\n', distortion_matrix_left,end = '\n')
    print('right:', ret_right, '\n', intrinsic_matrix_right, '\n', distortion_matrix_right,end = '\n')

    print(len(imgp_L),len(imgp_R))

    retS, K1, K2, D2, R, T, E, F = stereo_camera_calibration(obj_L, imgp_L, imgp_R, intrinsic_matrix_left, distortion_matrix_left, intrinsic_matrix_right, distortion_matrix_right, (640,480))
    print('stereo:',retS,'\n', K1, '\n', K2, '\n', D2, '\n', R, '\n', T, '\n', E, '\n', F, '\n')