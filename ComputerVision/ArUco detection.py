#Procedimentos:
#1: Definir a origem e calibrar a posição do objeto na origem escolhida;
#2: Calibrar a profundidade (z) utilizando regressão linear (medida real x medida obtida);
#3: Calibrar a relação entre a variável z e as coordenadas x e y. Relacionar os valores de (x,z) e (y,z).

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
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)

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
# markerImage = cv.aruco.drawMarker(dictionary, 39, 200, markerImage, 1)

# cv.imwrite("marker39.png", markerImage)

#Create position data.txt
pos_data = open("pos_data.txt",'w')
pos_data.close()

#Create attitude data.txt
angle_data = open("angle_data.txt", 'w')
angle_data.close()

#Camera instance with thread
cap = WebcamVideoStream(src=0).start()
# cap = cv.VideoCapture(0)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1280)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1024)
# cap = cv.VideoCapture(0)
# addres = "https://192.168.42.129:8080/video"
# cap.open(addres)


#Create board object
# board = cv.aruco.GridBoard_create(2, 2, 0.037, 0.005, dictionary)

start_clock = time.perf_counter()

#Declare somes important variables
rvecs = None
tvecs = None

frame_id = 0

pos_x = []
pos_y = []
pos_z = []

offset_x = 0
offset_y = 0
offset_z = 0

#Set position calibration
calib_pos = True

#Set Kalman Filter
kalman_est = False

#Sample time
dt = 0.01
#System matrix and output matrix
F = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, dt, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, dt, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, dt, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, dt],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
#Initialization
x0 = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
P0 = np.eye(14)*10

#Covariances matrices
Q = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])*0.1


R = np.array([[0.1, 0, 0, 0, 0, 0, 0],
              [0, 0.1, 0, 0, 0, 0, 0],
              [0, 0, 0.1, 0, 0, 0, 0],
              [0, 0, 0, 100, 0, 0, 0],
              [0, 0, 0, 0, 100, 0, 0],
              [0, 0, 0, 0, 0, 100, 0],
              [0, 0, 0, 0, 0, 0, 100]])



while True:
    img = cap.read()
 
    img = cv.rotate(img, cv.ROTATE_180)
    #Convert frame to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #Frame count increment
    frame_id += 1

    #Set parameters for the marker tracking
    parameters = cv.aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10
    parameters.cornerRefinementWinSize = 10
    parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
    # parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0

    #Detect the markers in the image
    markerCorners, markerIDs, rejectedCandidates = cv.aruco.detectMarkers(gray, dictionary, parameters = parameters)
    #Refine detection
    # markerCorners, markerIDs, rejectedCandidates, recoveredIds = cv.aruco.refineDetectedMarkers(img, board, markerCorners,
                                                                                                # markerIDs, rejectedCandidates, mtx, dist)
    #Open position file to write data position
    pos_data = open("pos_data.txt", "a+")

    #Open attitude file to write estimation
    angle_data = open("angle_data.txt", "a+")

    # print('ID:', markerIDs)
    #Verify if there is some marker detected
    if np.all(markerIDs is not None):
        
        for i in range(0, len(markerCorners)):
            # print("Marker Detected")
            #Compute board's pose
            # pose, rvecs, tvecs = cv.aruco.estimatePoseBoard(markerCorners, markerIDs, board, mtx, dist, rvecs, tvecs)
            rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners[i], 0.066, mtx, dist)
            rvecs = np.reshape(rvecs, (3,1))
            tvecs = np.reshape(tvecs, (3,1))
            #Use Rodrigues formula to transform rotation vector into matrix
            R_matrix, _ = cv.Rodrigues(rvecs)
            #Rotation Matrix marker to camera
            R_marker = R_matrix.T

            R_flip = np.array([[1, 0, 0],[0, -1, 0], [0, 0, -1]])
            #Getting quaternions from rotation matrix
            r = sci.spatial.transform.Rotation.from_matrix(R_flip@R_marker)
            q = r.as_quat()
            # # print('Quaternion:')
            # # print(q)

            euler = r.as_euler('XYZ', degrees=True)
            phi = euler[0]
            theta = euler[1]
            psi = euler[2]

            #Attitude estimation
            # phi = 180*math.atan2(R_marker[2,1], R_marker[2,2])/math.pi
            # theta = 180*math.atan2(-R_matrix[2, 0], math.sqrt(R_matrix[2,1]**2 + R_matrix[2,2]**2))/math.pi
            # psi = 180*math.atan2(R_matrix[1,0], R_matrix[0,0])/math.pi 
            
            cv.aruco.drawAxis(img, mtx, dist, rvecs, tvecs, 0.066)

            cv.aruco.drawDetectedMarkers(img, markerCorners) 
            #Calibration position: In the beggining of the video, the script will get 40 early datas to compute the mean
            #measurements and so find offset values. Having the offset, we can calculate the relative origin.
            #Keep in mind that the marker's position was acquired with respect to camera frame.
            if calib_pos:

                pos_x.append(tvecs[0])
                pos_y.append(tvecs[1])
                pos_z.append(tvecs[2])

                if len(pos_x) == 20 and len(pos_y) == 20 and len(pos_z):
                    
                    offset_x = sum(pos_x)/len(pos_x)
                    offset_y = sum(pos_y)/len(pos_y)
                    # offset_z = sum(pos_z)/len(pos_z)
                    
                    print(offset_x)
                    print(offset_y)
                    # print(offset_z)

                    calib_pos = False

            else:
                xf = 0
                yf = 0
                zf = 0

            # offset_x = -0.87540023
            # offset_y = -0.2311414
            
            zf = float(tvecs[2])*0.316 + 0.002
            xf = float(tvecs[0]) - offset_x
            yf = float(tvecs[1]) - offset_y


            #Kalman Filter

            if calib_pos is False:

                if kalman_est:
                    #X estimation
                    #Update
                    x_est_e =  np.dot(F, x0)
                    P_k = F@P0@F.T + Q
                    K = P_k@H.T@inv(H@P_k@H.T+R)

                    #Assimilation
                    inov = np.array([[xf],[yf],[zf],[q[0]],[q[1]],[q[2]],[q[3]]]) - H@x_est_e
                    x_est_a = x_est_e + K@inov
                    P0 = P_k - K@H@P_k
                    x0 = x_est_a

                    q_est = np.array([[float(x_est_a[6]), float(x_est_a[7]), float(x_est_a[8]), float(x_est_a[9])]])
                    r_est = sci.spatial.transform.Rotation.from_quat(q_est)
                    euler_est = r_est.as_euler('XYZ', degrees=True)
                    phi_est = euler_est[0,0]
                    theta_est = euler_est[0,1]
                    psi_est = euler_est[0,2]

                    #Print position values in frame
                    cv.putText(img, "X:"+str(np.round(float(x_est_a[0]), 4)), (50,400), font, 1, (0,0,255), 2)
                    cv.putText(img, "Y:"+str(np.round(float(x_est_a[1]), 4)), (200,400), font, 1, (0,255,0), 2)
                    cv.putText(img, "Z:"+str(np.round(float(x_est_a[2]), 4)), (350,400), font, 1, (255,0,0), 2)
                    cv.putText(img, "Phi:"+str(np.round(float(phi_est), 2)), (10,120), font, 1, (0,0,255), 2)
                    cv.putText(img, "Theta:"+str(np.round(float(theta_est), 2)), (10,160), font, 1, (0,255,0), 2)
                    cv.putText(img, "Psi:"+str(np.round(float(psi_est), 2)), (10,200), font, 1, (255,0,0), 2)

                    #Write position data in pos_data.txt
                    pos_data.write("{:.4f} , ".format(float(x_est_a[0])) + "{:.4f} , ".format(float(x_est_a[1])) + "{:.4f} , ".format(float(x_est_a[2])) + str(time.process_time()-start_clock) + "\n")
                    #Write attitude data in file
                    angle_data.write("{:.2f} , ".format(float(x_est_a[6])) + "{:.2f} , ".format(float(x_est_a[7])) + "{:.2f} , ".format(float(x_est_a[8])) + str(time.process_time()-start_clock) + "\n")
                
                else:
                    #Print position values in frame
                    cv.putText(img, "X:"+str(np.round(float(xf), 4)), (10,400), font, 1, (0,0,255), 2)
                    cv.putText(img, "Y:"+str(np.round(float(yf), 4)), (100,400), font, 1, (0,255,0), 2)
                    cv.putText(img, "Z:"+str(np.round(float(zf), 4)), (200,400), font, 1, (255,0,0), 2)
                    cv.putText(img, "Phi:"+str(np.round(float(phi), 2)), (10,120), font, 1, (0,0,255), 2)
                    cv.putText(img, "Theta:"+str(np.round(float(theta), 2)), (10,160), font, 1, (0,255,0), 2)
                    cv.putText(img, "Psi:"+str(np.round(float(psi), 2)), (10,200), font, 1, (255,0,0), 2)

                    #Write position data in pos_data.txt
                    pos_data.write("{:.4f} , ".format(float(xf)) + "{:.4f} , ".format(float(yf)) + "{:.4f} , ".format(float(zf)) + str(time.process_time()-start_clock) + "\n")
                    #Write attitude data in file
                    angle_data.write("{:.2f} , ".format(float(phi)) + "{:.2f} , ".format(float(theta)) + "{:.2f} , ".format(float(psi)) + str(time.process_time()-start_clock) + "\n")


    #Compute FPS
    elapsed_time = time.time() - start_time
    fps = int(frame_id / elapsed_time)
    #Print FPS on the screen
    cv.putText(img, "FPS:" + str(fps), (10, 80), font, 1, (255,255,255), 1)
    cv.imshow('img', img)
    
    key = cv.waitKey(10)
    if key == ord('n') or key == ord('p'):
        break

pos_data.close()
cap.stop()
# cap.release()
cv.destroyAllWindows()


# plt.style.use("fivethirtyeight")
# fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(10,10), sharex=True)
# x, y, z, time = [], [], [], []

# data = open("pos_data.txt", "r").read()
# lines = data.split('\n')

# for line in lines:
#     if len(line)>1:
#         xs, ys, zs, times = line.split(' , ')
#         x.append(float(xs))
#         y.append(float(ys))
#         z.append(float(zs))
#         time.append(float(times))


# ax.plot(time, x, 'r', label = r'$x(t)$', linewidth = 1)
# ax2.plot(time, y, 'g', label = r'$y(t)$', linewidth=1)
# ax3.plot(time, z, 'b', label = r'$z(t)$', linewidth=1)
# ax3.set_xlabel('Time (s)')

# ax.set_ylabel(r'$x$ (meters)')
# ax2.set_ylabel(r'$y$ (meters)')
# ax3.set_ylabel(r'$z$ (meters)')
# ax.set_title('Posição')

# ax.set_ylim([-1, 1])
# ax2.set_ylim([-1, 1])
# ax3.set_ylim([-1, 1])
# ax.legend()
# ax2.legend()
# ax3.legend()

# plt.tight_layout()
# plt.show()


