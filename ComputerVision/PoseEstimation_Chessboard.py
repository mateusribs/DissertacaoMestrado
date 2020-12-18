import numpy as np
import cv2 as cv
from imutils.video import WebcamVideoStream
import glob
import time
import math 
# Load previously saved data
path = './camera_data/camera_calibration.npz'
npzfile = np.load(path)
mtx = npzfile[npzfile.files[0]]
dist = npzfile[npzfile.files[1]]
print(mtx ,dist)

#Setup de fonte
font = cv.FONT_HERSHEY_PLAIN
start_time = time.time()

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*0.26

axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)

cap = WebcamVideoStream(src=0).start()
frame_id = 0

while True:
    img = cap.read()
    frame_id += 1
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9,6),None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs,_ = cv.solvePnPRansac(objp, corners2, mtx, dist)
        R_matrix,_ = cv.Rodrigues(rvecs)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
        cv.putText(img, "Euler Angles" + str(R_matrix), (10, 150), font, 1, (255, 255, 255), 1)

    elapsed_time = time.time() - start_time
    fps = int(frame_id / elapsed_time)
    cv.putText(img, "FPS:" + str(fps), (10, 80), font, 1, (255,255,255), 1)
    
    cv.imshow('img',img)
        
    key = cv.waitKey(10)
    if key == 27:
        break

cap.stop()
cv.destroyAllWindows()