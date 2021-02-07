import cv2 as cv
import numpy as np
from numpy.linalg import inv, det
import scipy as sci
from scipy.spatial.transform import Rotation as R
from imutils.video import WebcamVideoStream
import matplotlib.pyplot as plt
import glob
import sys
import time
import math 


#Load the predefinied dictionary
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_APRILTAG_36h10)

font = cv.FONT_HERSHEY_PLAIN

# Load previously saved calibration data
path = './camera_data/camera_calibration.npz'
npzfile = np.load(path)
#Camera Matrix
mtx = npzfile[npzfile.files[0]]
#Distortion Matrix
dist = npzfile[npzfile.files[1]]

cap = WebcamVideoStream(src=0).start()

#Object points
obj_points = np.array([[-0.066/2, 0.066/2, 0.0],
                       [0.066/2, 0.066/2, 0.0],
                       [0.066/2, -0.066/2, 0.0],
                       [-0.066/2, -0.066/2, 0.0]])

rvec = None
tvec = None
error = np.array((1,1), dtype='float32')

while True:
    frame = cap.read()
    frame = cv.rotate(frame, cv.ROTATE_180)
    #Convert to gray scale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #Set parameters to marker's detection
    parameters = cv.aruco.DetectorParameters_create()
    #Detect marker corners and ID's, as also rejected candidates
    markerCorners, markerIDs, rejectedCandidates = cv.aruco.detectMarkers(gray, dictionary, parameters = parameters)

    if np.all(markerIDs is not None):
        for i in range(0, len(markerCorners)):
            corners = np.resize(markerCorners, (4,2))
            print(corners.shape)
            print(obj_points.shape)
            _, rvecs, tvecs, error = cv.solvePnPGeneric(obj_points, corners, mtx, dist, flags=cv.SOLVEPNP_IPPE_SQUARE, reprojectionError=error)
            print(error)
            # rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners[i], 0.066, mtx, dist)
            R_matrix, _ = cv.Rodrigues(rvecs[0])
            # cv.aruco.drawAxis(frame, mtx, dist, rvecs, tvecs, 0.066)
            # project 3D points to image plane
            axis = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]).reshape(-1,3)
            imgpts, jac = cv.projectPoints(axis, rvecs[0], tvecs[0], mtx, dist)
            imgpts = imgpts.astype(np.int)

            imgpoint = tuple(corners[0].ravel())
            cv.line(frame, imgpoint, tuple(imgpts[0].ravel()), (255,0,0), 3)
            cv.line(frame, imgpoint, tuple(imgpts[1].ravel()), (0,255,0), 3)
            cv.line(frame, imgpoint, tuple(imgpts[2].ravel()), (0,255,255), 3)
            

            r = sci.spatial.transform.Rotation.from_matrix(R_matrix)
            q = r.as_quat()
            # # print('Quaternion:')
            # # print(q)

            euler = r.as_euler('XYZ', degrees=True)
            phi = euler[0]
            theta = euler[1]
            psi = euler[2]

            cv.putText(frame, "Phi:"+str(np.round(float(phi), 2)), (10,120), font, 1, (0,0,255), 2)
            cv.putText(frame, "Theta:"+str(np.round(float(theta), 2)), (10,160), font, 1, (0,255,0), 2)
            cv.putText(frame, "Psi:"+str(np.round(float(psi), 2)), (10,200), font, 1, (255,0,0), 2)

    cv.imshow('frame',frame)

    key = cv.waitKey(10)
    if key == ord('n') or key == ord('p'):
        break

cv.destroyAllWindows()
cap.stop()