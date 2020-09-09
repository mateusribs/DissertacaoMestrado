import cv2 as cv
import numpy as np
from imutils.video import WebcamVideoStream
import glob
import time
import math 

#Load the predefinied dictionary
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

# Load previously saved calibration data
path = './camera_data/camera_calibration.npz'
npzfile = np.load(path)
#Camera Matrix
mtx = npzfile[npzfile.files[0]]
#Distortion Matrix
dist = npzfile[npzfile.files[1]]
print(mtx ,dist)

#Font setup
font = cv.FONT_HERSHEY_PLAIN
start_time = time.time()

# #Generate the marker
# markerImage = np.zeros((200,200), dtype=np.uint8)
# markerImage = cv.aruco.drawMarker(dictionary, 33, 200, markerImage, 1)

# cv.imwrite("marker33.png", markerImage)

#Camera instance with thread
cap = WebcamVideoStream(src=0).start()

frame_id = 0

while True:
    img = cap.read()
    frame_id += 1

    #Initialize the detector parameters using defaults values
    parameters = cv.aruco.DetectorParameters_create()

    #Detect the markers in the image
    markerCorners, markerIDs, rejectedCandidates = cv.aruco.detectMarkers(img, dictionary, parameters=parameters)

    if np.all(markerIDs != None):
        rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners, 0.05, mtx, dist)

        for i in range(0, markerIDs.size):
            cv.aruco.drawAxis(img, mtx, dist, rvecs[i], tvecs[i], 0.1)

        cv.aruco.drawDetectedMarkers(img, markerCorners)    


    cv.imshow('img', img)
        
    key = cv.waitKey(10)
    if key == 27:
        break

cap.stop()
cv.destroyAllWindows()

