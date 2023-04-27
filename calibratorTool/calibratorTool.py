import numpy as np
import cv2
import glob

calibration_dir = 'left_pic/'  #标定图片存储路径
calibration_files = calibration_dir+'*.png' #标定图片文件后缀
image_file = glob.glob(calibration_files) #将所有路径下png格式文件遍历并存到image_file列表中，返回列表
#print(image_file) #列表值
chessboard_size = (11,8) #棋盘格的横纵corner数量,标定板规格
width = 11
height = 8
square_size = 25 #棋盘格的边长，当前使用的菲林标定板为25mm

objp = np.zeros((height*width,3),np.float32)
objp[:,:2] = np.mgrid[0:width,0:height].T.reshape(-1,2)
objp *= square_size
#print(objp)
imgpointsL = []
imgpointsR = []
objpoints = []
for fname in image_file: #遍历列表中的图片
    img = cv2.imread(fname) #图片读取

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #图片转为灰度图

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None) #findChessboardCorners函数，返回布尔值和检测到的角点坐标 格式为1行2列n层的矩阵
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)) #亚像素角点检测精确化，对检测出的角点进行精确化处理，返回精确角点坐标

        img_corner_identify = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret) #在原图像上标识corner所在位置
        '''
        cv2.imshow(fname, img_corner_identify)
        cv2.waitKey(1000)
        cv2.destroyAllWindows() #显示图像，1000ms后自动关闭所有窗口
        '''
        imgpointsL.append(corners2)
        objpoints.append(objp)
    print(fname)
retL, K1, D1, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, gray.shape[::-1], None, None) #转为(width,height) 使用::-1 ret,内参，畸变，旋转，平移矩阵
print(objpoints[0].shape,imgpointsL[0].shape)
print(len(objpoints),len(imgpointsL))

print(retL,'\n',K1,'\n', D1,'\n' ,rvecsL,'\n', tvecsL )

