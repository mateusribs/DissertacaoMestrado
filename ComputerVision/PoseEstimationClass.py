import numpy as np
import cv2 as cv
from imutils.video import WebcamVideoStream
import glob
import time
import math 

class PoseEstimation():
    def __init__(self, mtx, dist):

        self.mtx = mtx
        self.dist = dist


    def detect_contourn(self, image, color):
        
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        if color == "Red":
            #Define the limits in HSV variables
            self.lower = np.array([0, 35, 225])
            self.upper = np.array([0, 255, 255])
        if color == "Green":
            #Define the limits in HSV variables
            self.lower = np.array([48, 35, 225])
            self.upper = np.array([65, 255, 255])
        if color == "Blue":
            #Define the limits in HSV variables
            self.lower = np.array([70, 35, 225])
            self.upper = np.array([120, 255, 255])
        if color == "Yellow":
            #Define the limits in HSV variables
            self.lower = np.array([20, 100, 100])
            self.upper = np.array([32, 220, 255])

        #Define threshold for red color
        mask = cv.inRange(hsv, self.lower, self.upper)
        #Create a kernel
        kernel = np.ones((5,5), np.uint8)
        #Apply opening process
        opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations = 1)
        #Find BLOB's contours
        cnts, _ = cv.findContours(opening.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

        return  cnts

    def center_mass_calculate(self, image, c):

        # Compute the center of the contour
        M = cv.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        _, radius = cv.minEnclosingCircle(c)
        cX = int(cX)
        cY = int(cY)
        center = (cX, cY)
        radius = int(radius)
        perimeter = cv.arcLength(c, True)
        #Compute the eccentricity
        metric = (4*math.pi*M["m00"])/perimeter**2
        if metric > 0.8:
            #Draw the contour and center of the shape on the image
            cv.drawContours(image, [c], -1, (0, 0, 0), 1)
            # cv.circle(image, center, radius, (0, 0, 0),1)
            cv.circle(image, center, 1, (0, 0, 0), -1)
            # cv.circle(image, (cX+radius, cY), 1, (0, 0, 0), 2)
            # cv.circle(image, (cX-radius, cY), 1, (0, 0, 0), 2)
            # cv.circle(image, (cX, cY+radius), 1, (0, 0, 0), 2)
            # cv.circle(image, (cX, cY-radius), 1, (0, 0, 0), 2)
                          
        return cX, cY, radius

    def draw(self, image, imgpoints, imgpts):
        imgpoint = tuple(imgpoints[0].ravel())
        img = cv.line(image, imgpoint, tuple(imgpts[0].ravel()), (255,0,0), 3)
        img = cv.line(image, imgpoint, tuple(imgpts[1].ravel()), (0,255,0), 3)
        img = cv.line(image, imgpoint, tuple(imgpts[2].ravel()), (0,255,255), 3)
        return img
    
    def get_element_vector(self, f1, f2, c1, c2):
        #Where f1 is frameA, f2 is frameB
        #c1 is the coordinate, let x = 0, y = 1, z = 2
        vec = []
        for i in range(np.shape(f1)[0]):
            cc = f2[i, c1]*f1[i, c2]
            vec.append(cc)
        return np.sum(vec)

    def get_element_A(self, f1, c1, c2):
        A = []
        for i in range(np.shape(f1)[0]):
            cc = f1[i, c1]*f1[i, c2]
            A.append(cc)
        return np.sum(A)

    def get_element_last(self, f1, c1):
        last = []
        for i in range(np.shape(f1)[0]):
            cc = f1[i, c1]
            last.append(cc)
        return np.sum(last)

    def get_transform_frame(self, f1, f2):
        matrix = np.zeros((3,4))
        for i in range(3):
            for j in range(3):
                matrix[i, j] = self.get_element_vector(f1, f2, i, j)
                matrix[i, 3] = self.get_element_last(f2, i)

        A = np.zeros((4,4))
        for i in range(3):
            for j in range(3):
                A[i, j] = self.get_element_A(f1, i, j)

        for i in range(3):
            A[i,3] = self.get_element_last(f1, i)
            A[3, i] = self.get_element_last(f1, i)

        A[3,3] = np.shape(f1)[0]
        A_inv = np.linalg.inv(A)

        matrix = np.transpose(matrix)

        T = np.dot(A_inv, matrix)
        T = np.transpose(T)
        last_row = np.array([0,0,0,1]).reshape(1,4)
        T = np.concatenate((T, last_row), axis=0)
        
        return T

    def get_pose(self, image, objpoints, imgpoints, mtx, dist):
        
        axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(-1,3)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs,_ = cv.solvePnPRansac(objpoints, imgpoints, mtx, dist, iterationsCount=5)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        imgpts = imgpts.astype(np.int)
        img = self.draw(image, imgpoints, imgpts)
        R_matrix, _ = cv.Rodrigues(rvecs)
        
        return rvecs, tvecs, R_matrix, image