# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:07:14 2020

@author: Mateus Ribeiro
"""
import numpy as np
import cv2
import glob

workingdir="C:/Users/Mateus Ribeiro/Documents/Estudos/Mestrado/Visão Computacional/Detecção de Objetos"
savedir="camera_data/"

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp = objp*0.26
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('calibration/*.JPG')
path = './camera_data/camera_calibration.npz'

for fname in images:
    img = cv2.imread(fname)
    print(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    img1=img

print(">==> Starting calibration")
ret, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

h,  w = img1.shape[:2]
print("Image Width, Height")
print(w, h)
#if using Alpha 0, so we discard the black pixels from the distortion.  this helps make the entire region of interest is the full dimensions of the image (after undistort)
#if using Alpha 1, we retain the black pixels, and obtain the region of interest as the valid pixels for the matrix.
#i will use Apha 1, so that I don't have to run undistort.. and can just calculate my real world x,y
newcam_mtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, dist, (w,h), 1, (w,h))

inverse = np.linalg.inv(newcam_mtx)

np.savez(path, cam_mtx, dist, rvecs, tvecs, newcam_mtx, roi, inverse)

print(">==> Calibration ended")

# undistort
undst = cv2.undistort(img1, cam_mtx, dist, None, newcam_mtx)

# crop the image
#x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
#cv2.circle(dst,(308,160),5,(0,255,0),2)
cv2.imshow('img1', img1)
cv2.waitKey(5000)      
cv2.destroyAllWindows()
cv2.imshow('img1', undst)
cv2.waitKey(5000)      
cv2.destroyAllWindows()