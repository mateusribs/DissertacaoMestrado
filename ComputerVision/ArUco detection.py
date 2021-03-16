#Procedimentos:
#1: Definir a origem e calibrar a posição do objeto na origem escolhida
#2: Calibrar a profundidade (z) utilizando regressão linear (medida real x medida obtida)
#3: Calibrar a relação entre a variável z e as coordenadas x e y. Relacionar os valores de (x,z) e (y,z).

import cv2 as cv
from threading import Thread
import numpy as np
from numpy.linalg import inv, det
import scipy as sci
from scipy.spatial.transform import Rotation as R
from imutils.video import WebcamVideoStream
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial
import collections
import glob
import sys
import time
import math
from numpy import cos as c
from numpy import sin as s

from vpython import *

#Calibration position: In the beggining of the video, the script will get 40 early datas to compute the mean
#measurements and so find offset values. Having the offset, we can calculate the relative origin.
#Keep in mind that the marker's position was acquired with respect to camera frame.

isReceive = False
isRun = True
gx = 0.0
gy = 0.0
gz = 0.0
ax = 0.0
ay = 0.0
az = 0.0
cx = 0.0
cy = 0.0
cz = 0.0
dt = 0.1
g = 9.81
q_ant = np.array([1.0, 0.0, 0.0, 0.0])

x_ant_ori = np.array([[0], [0], [0], [0], [0], [0]])
P_ant_ori = np.eye(6)*1000

x_ant_pos = np.array([[0], [0], [0], [0], [0], [0]])
P_ant_pos = np.eye(6)*1000

def EKF_Accel_Camera(x_ant, P_ant, accel, y, dt):
    
    # Q ----> Process Noise Covariance
    # C ----> Control Noise Covariance
    # R  ---> Measurement Noise Covariance

    F = np.array([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    V = np.array([[0.5*dt**2, 0, 0],
                  [0, 0.5*dt**2, 0],
                  [0, 0, 0.5*dt**2],
                  [dt, 0, 0],
                  [0, dt, 0],
                  [0, 0, dt]])

    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])

    Q = np.eye(6)
    R = np.eye(3)

    #Process Equations
    x = F@x_ant + V@accel

    #State covariance matrix
    P = F@P_ant@F.T + Q

    #Kalman Gain and Inovation
    K = P@H.T@inv(H@P@H.T + R)
    z = y - H@x

    #Update step
    x_est = x + K@z
    P_est = P - K@H@P

    return x_est, P_est


def KF_Orientation(x_ant, P_ant, y, dt):

    A = np.array([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    
    B = np.array([[]])

    H = np.eye(6)

    Q = np.array([[10, 0, 0, 0, 0, 0],
                  [0, 10, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    
    R = np.array([[.1, 0, 0, 0, 0, 0],
                  [0, .1, 0, 0, 0, 0],
                  [0, 0, .01, 0, 0, 0],
                  [0, 0, 0, .01, 0, 0],
                  [0, 0, 0, 0, .01, 0],
                  [0, 0, 0, 0, 0, .01]])
    
    #Prediction
    x_p = A@x_ant
    P_p = A@P_ant@A.T + Q

    #Kalman Gain and Inovation
    K = P_p@H.T@inv(H@P_p@H.T + R)
    inov = y - H@x_p

    #Update
    x_est = x_p + K@inov
    P_est = P_p - K@H@P_p

    return x_est, P_est
    

def KF_Position(x_ant, P_ant, u, y, dt):

    A = np.array([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    
    B = np.array([[dt**2/2, 0, 0],
                  [0, dt**2/2, 0],
                  [0, 0, dt**2/2],
                  [dt, 0, 0],
                  [0, dt, 0],
                  [0, 0, dt]])

    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])

    Q = np.array([[100, 0, 0, 0, 0, 0],
                  [0, 100, 0, 0, 0, 0],
                  [0, 0, 100, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    
    R = np.array([[100, 0, 0],
                  [0, 100, 0],
                  [0, 0, 100]])

    
    #Prediction
    x_p = A@x_ant + B@u
    P_p = A@P_ant@A.T + Q

    #Kalman Gain and Inovation
    K = P_p@H.T@inv(H@P_p@H.T + R)
    inov = y - H@x_p

    #Update
    x_est = x_p + K@inov
    P_est = P_p - K@H@P_p


    return x_est, P_est


def quat_product(a, b):
    k1 = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
    k2 = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
    k3 = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
    k4 = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]

    k = np.array([k1, k2, k3, k4])
    
    return k


def MadgwickUpdate_OnlyAccel(gx, gy, gz, ax, ay, az, cx, cy, cz, dt, q_ant):

    q0 = q_ant[0]
    q1 = q_ant[1]
    q2 = q_ant[2]
    q3 = q_ant[3]
    beta = 0.5

    # Rate of change of quaternion from gyroscope
    qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
    qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
    qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
    qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

    # Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
    if ax !=0 and ay != 0 and az != 0:

        # Normalise accelerometer measurement
        recipNorm = 1/(math.sqrt(ax * ax + ay * ay + az * az))
        ax *= recipNorm
        ay *= recipNorm
        az *= recipNorm

        print('ax: {0}, ay: {1}, az: {2}'.format(ax, ay, az))

        # Auxiliary variables to avoid repeated arithmetic
        _2q0 = 2.0 * q0
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _4q0 = 4.0 * q0
        _4q1 = 4.0 * q1
        _4q2 = 4.0 * q2
        _4q3 = 4.0 * q3
        _8q1 = 8.0 * q1
        _8q2 = 8.0 * q2
        q0q0 = q0 * q0
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3

        f1 = 2*(q1*q3 - q0*q2) - ax
        f2 = 2*(q0*q1 + q2*q3) - ay
        f3 = 2*(0.5 - q1q1 - q2q2) - az
        f4 = 2*(0.5 - q2q2 - q3q3) - cx
        f5 = 2*(q1*q2 - q0*q3) - cy
        f6 = 2*(q0*q2 + q1*q3) - cz

        print('cx: {0}, cy: {1}, cz: {2}'.format(cx, cy, cz))

        # Gradient decent algorithm corrective step
        s0 = -_2q2*f1 + _2q1*f2 - _2q3*f5 + _2q2*f6
        s1 = _2q3*f1 + _2q0*f2 - _4q1*f3 + _2q2*f5 + _2q3*f6
        s2 = -_2q0*f1 + _2q3*f2 - _4q2*f3 - _4q2*f4 + _2q1*f5 + _2q0*f6
        s3 = _2q1*f1 + _2q2*f2 - _4q3*f4 - _2q0*f5 + _2q1*f6
        recipNorm = 1/ (math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)) ## normalise step magnitude
        s0 *= recipNorm
        s1 *= recipNorm
        s2 *= recipNorm
        s3 *= recipNorm

        # Apply feedback step
        qDot1 -= beta * s0
        qDot2 -= beta * s1
        qDot3 -= beta * s2
        qDot4 -= beta * s3


    # Integrate rate of change of quaternion to yield quaternion
    q0 += qDot1 * dt
    q1 += qDot2 * dt
    q2 += qDot3 * dt
    q3 += qDot4 * dt

    # Normalise quaternion
    recipNorm = 1 / (math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3))
    q0 *= recipNorm
    q1 *= recipNorm
    q2 *= recipNorm
    q3 *= recipNorm

    q = np.array([q0, q1, q2, q3])

    # print('q0: {0}, q1: {1}, q2: {2}, q3: {3}'.format(q[0], q[1], q[2], q[3]))p

    return q


def computeAngles(q0, q1, q2, q3):

    roll = 180*math.atan2(q0*q1 + q2*q3, 0.5 - q1*q1 - q2*q2)/math.pi
    pitch = 180*math.asin(-2.0 * (q1*q3 - q0*q2))/math.pi
    yaw = 180*math.atan2(q1*q2 + q0*q3, 0.5 - q2*q2 - q3*q3)/math.pi

    return roll, pitch, yaw


def getData():
    time.sleep(1.0)
    serialConnection.reset_input_buffer()

    while(isRun):

        global isReceive
        global roll, pitch, yaw
        global q0, q1, q2, q3
        global accel
        global ax, az, ay, gx, gy, gz

        data = serialConnection.readline()
        data = str(data, 'utf-8')
        splitdata = data.split(',')
        q0 = float(splitdata[0])
        q1 = float(splitdata[1])
        q2 = float(splitdata[2])
        q3 = float(splitdata[3])
        ax = float(splitdata[4]) 
        ay = float(splitdata[5]) 
        az = float(splitdata[6])
        gx = float(splitdata[7])
        gy = float(splitdata[8])
        gz = float(splitdata[9]) 

        accel = np.array([[ax], [ay], [az]])

        roll, pitch, yaw = computeAngles(q0, q1, q2, q3)

        isReceive = True



serialPort = 'COM5'
baudRate = 9600

try:
    serialConnection = serial.Serial(serialPort, baudRate)
except:
    print('Cannot connect to the port')


thread = Thread(target=getData)
thread.start()

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

#Create position data.txt
pos_data = open("pos_data.txt",'w')
pos_data.close()

#Create attitude data.txt
angle_data = open("angle_data.txt", 'w')
angle_data.close()

#Camera instance with thread
cap = WebcamVideoStream(src=0).start()
# cap = cv.VideoCapture(0)

#Create board object
# board = cv.aruco.GridBoard_create(2, 2, 0.037, 0.005, dictionary)

start_clock = time.perf_counter()

#Declare somes important variables
rvecs = None
tvecs = None

pos_calib_obj = True

offset_xo = 0
offset_yo = 0
offset_zo = 0

frame_id = 0

roll_o = []
pitch_o = []
yaw_o = []

roll_c = []
pitch_c = []
yaw_c = []

pos_xo = []
pos_yo = []
pos_zo = []

roll_obj = 0
pitch_obj = 0
yaw_obj = 0

alfa = 0
beta = 0
gama = 0

while True:
    img = cap.read()
 
    # img = cv.rotate(img, cv.ROTATE_180)
    #Convert frame to gray scale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #Frame count increment
    frame_id += 1

    #Set parameters for the marker tracking
    parameters = cv.aruco.DetectorParameters_create()
    parameters.minMarkerPerimeterRate = 0.1
    # parameters.minDistanceToBorder = 6
    # parameters.adaptiveThreshConstant = 25
    parameters.cornerRefinementWinSize = 3
    parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
    # parameters.cornerRefinementMaxIterations = 10
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
    if markerIDs is not None and len(markerIDs)==2:
        
        for i in range(0, len(markerCorners)):
            # print("Marker Detected")
            #Compute board's pose
            
            #Reference marker
            if markerIDs[i]==10:
                # pose, rvecs, tvecs = cv.aruco.estimatePoseBoard(markerCorners, markerIDs, board, mtx, dist, rvecs, tvecs)
                rvec_ref, tvec_ref, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners[i], 0.066, mtx, dist)
                rvec_ref = np.reshape(rvec_ref, (3,1))
                tvec_ref = np.reshape(tvec_ref, (3,1))

                #Use Rodrigues formula to transform rotation vector into matrix
                #Pose marker w.r.t camera reference frame
                R_rc, _ = cv.Rodrigues(rvec_ref)
    
                #Homogeneous Transformation Fixed Frame to Camera Frame
                last_col = np.array([[0, 0, 0, 1]])
                T_rc = np.concatenate((R_rc, tvec_ref), axis=1)
                T_rc = np.concatenate((T_rc, last_col), axis=0)
                #Homegeneous Transformation Camera Frame to Fixed Frame
                T_cr = np.linalg.inv(T_rc)

                #Euler angles
                # r_ref = sci.spatial.transform.Rotation.from_matrix(T_cr[0:3, 0:3])
                # euler_ref = r_ref.as_euler('ZYX', degrees=True)
                # phi_ref = euler_ref[0]
                # theta_ref = euler_ref[1]
                # psi_ref = euler_ref[2]

                # print("phi:", phi_ref)
                # print("theta:", theta_ref)
                # print("psi:", psi_ref)

                #Attitude estimation
                # phi = 180*math.atan2(R_marker[2,1], R_marker[2,2])/math.pi
                # theta = 180*math.atan2(-R_matrix[2, 0], math.sqrt(R_matrix[2,1]**2 + R_matrix[2,2]**2))/math.pi
                # psi = 180*math.atan2(R_matrix[1,0], R_matrix[0,0])/math.pi 
                
                cv.aruco.drawAxis(img, mtx, dist, rvec_ref, tvec_ref, 0.033)
                cv.aruco.drawDetectedMarkers(img, markerCorners)

            #Object marker
            if markerIDs[i]==4:

                # pose, rvecs, tvecs = cv.aruco.estimatePoseBoard(markerCorners, markerIDs, board, mtx, dist, rvecs, tvecs)
                rvec_obj, tvec_obj, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners[i], 0.066, mtx, dist)
                rvec_obj = np.reshape(rvec_obj, (3,1))
                tvec_obj = np.reshape(tvec_obj, (3,1))

                #Use Rodrigues formula to transform rotation vector into matrix
                R_dc, _ = cv.Rodrigues(rvec_obj)

                #Homogeneous Transformation Object Frame to Camera Frame
                T_dc = np.concatenate((R_dc, tvec_obj), axis=1)
                T_dc = np.concatenate((T_dc, last_col), axis=0)

                #Homogeneous Transformation Object Frame to Fixed Frame
                T_dr = T_cr@T_dc

                #Getting quaternions from rotation matrix
                r_obj = sci.spatial.transform.Rotation.from_matrix(T_dr[0:3, 0:3])
                q_obj = r_obj.as_quat()
                # # print('Quaternion:')
                # # print(q)

                roll_obj, pitch_obj, yaw_obj = computeAngles(q_obj[3], q_obj[1], q_obj[0], -q_obj[2])
                roll_obj -= 8.2
                pitch_obj += 7.2

                # euler_obj = r_obj.as_euler('ZYX', degrees=True)
                # phi_obj = euler_obj[0]
                # theta_obj = euler_obj[1]
                # psi_obj = euler_obj[2]

                # print("phi:", phi_obj)
                # print("theta:", theta_obj)
                # print("psi:", psi_obj)

                #Position correction    

                # xf_obj = 0.755*float(T_dr[0,3]) - 0.193 
                # yf_obj = 1.06*float(T_dr[1,3]) + 0.018 
                # zf_obj = 0.709*float(T_dr[2,3]) + 0.0687

                xf_obj = 0.774*float(T_dr[0,3]) - 0.129
                yf_obj = -0.349*float(T_dr[1,3])**2 + 1.09*float(T_dr[1,3]) + 0.129
                zf_obj = 0.767*float(T_dr[2,3]) - 0.0694

                # if pos_calib_obj:
                #     #Position correction
                #     pos_xo.append(xf_obj)
                #     pos_yo.append(yf_obj)
                #     pos_zo.append(zf_obj)

                #     if len(pos_xo) == 100 and len(pos_yo) == 100 and len(pos_zo)==100:
                                    
                #         # print('X: {0}, Y: {1}, Z: {2}'.format(np.average(pos_xo), np.average(pos_yo), np.average(pos_zo)))

                #         pos_xo=[]
                #         pos_yo=[]
                #         pos_zo=[]

                cv.aruco.drawAxis(img, mtx, dist, rvec_obj, tvec_obj, 0.033)
                print(tvec_obj)
                cv.aruco.drawDetectedMarkers(img, markerCorners)


                modulo = np.sqrt(q_obj[0]**2 + q_obj[1]**2 + q_obj[2]**2 + q_obj[3]**2)
                #Print position values in frame
                cv.putText(img, "X:"+str(np.round(float(xf_obj), 4)), (80,600), font, 1, (0,0,0), 2)
                cv.putText(img, "Y:"+str(np.round(float(yf_obj), 4)), (180,600), font, 1, (0,0,0), 2)
                cv.putText(img, "Z:"+str(np.round(float(zf_obj), 4)), (280,600), font, 1, (0,0,0), 2)
                
                cv.putText(img, "Orientacao Estimada por Camera:", (10, 200), font, 1, (255, 255, 255), 2)
                cv.putText(img, "Phi:"+str(np.round(float(roll_obj), 2)), (10,220), font, 1, (0,0,255), 2)
                cv.putText(img, "Theta:"+str(np.round(float(pitch_obj), 2)), (10,240), font, 1, (0,255,0), 2)
                cv.putText(img, "Psi:"+str(np.round(float(yaw_obj), 2)), (10,260), font, 1, (255,0,0), 2)
                # cv.putText(img, "q0:"+str(np.round(float(q_obj[3]), 3)), (500,120), font, 1, (255,255,255), 2)
                # cv.putText(img, "q1:"+str(np.round(float(q_obj[0]), 3)), (500,140), font, 1, (255,255,255), 2)
                # cv.putText(img, "q2:"+str(np.round(float(q_obj[1]), 3)), (500,160), font, 1, (255,255,255), 2)
                # cv.putText(img, "q3:"+str(np.round(float(q_obj[2]), 3)), (500,180), font, 1, (255,255,255), 2)
                # cv.putText(img, "Modulo:" + str(np.round(float(modulo), 3)), (500, 210), font, 1, (255, 0, 0),2)

            
            #Data sensor
            if isReceive:

                # cv.putText(img, "q0:"+str(np.round(float(q0), 3)), (500,320), font, 1, (0,0,0), 2)
                # cv.putText(img, "q1:"+str(np.round(float(q1), 3)), (500,340), font, 1, (0,0,0), 2)
                # cv.putText(img, "q2:"+str(np.round(float(q2), 3)), (500,360), font, 1, (0,0,0), 2)
                # cv.putText(img, "q3:"+str(np.round(float(q3), 3)), (500,380), font, 1, (0,0,0), 2)

                cv.putText(img, "Orientacao Estimada por IMU:", (10, 300), font, 1, (255,255,255), 2)
                cv.putText(img, "Roll:"+str(np.round(float(roll), 3)), (10,320), font, 1, (0,0,255), 2)
                cv.putText(img, "Pitch:"+str(np.round(float(pitch), 3)), (10,340), font, 1, (0,255,0), 2)
                cv.putText(img, "Psi:"+str(np.round(float(yaw), 3)), (10,360), font, 1, (255,0,0), 2)

                #########
                roll_c.append(roll_obj)
                pitch_c.append(pitch_obj)
                yaw_c.append(yaw_obj)

                if len(roll_c) == 100 and len(pitch_c) == 100 and len(yaw_c)==100:

                    print('Roll_C: {0}, Pitch_C: {1}, Yaw_C: {2}'.format(np.average(roll_c), np.average(pitch_c), np.average(yaw_c)))

                    roll_c=[]
                    pitch_c=[]
                    yaw_c=[]

                ######
                roll_o.append(roll)
                pitch_o.append(pitch)
                yaw_o.append(yaw)

                if len(roll_o) == 100 and len(pitch_o) == 100 and len(yaw_o)==100:

                    print('Roll: {0}, Pitch: {1}, Yaw: {2}'.format(np.average(roll_o), np.average(pitch_o), np.average(yaw_o)))


                    roll_o=[]
                    pitch_o=[]
                    yaw_o=[]

                #Kalman Filter - Orientation Estimation

                y_ori = np.array([[roll],[pitch],[yaw_obj],[gx],[gy],[gz]])
                x_est, P_est = KF_Orientation(x_ant_ori, P_ant_ori, y_ori, dt)
                x_ant_ori = x_est
                P_ant_ori = P_est

                roll_est = x_est[0]
                pitch_est = x_est[1]
                yaw_est = x_est[2]

                cv.putText(img, "Orientacao Estimada por Filtro:", (10, 400), font, 1, (255,255,255), 2)
                cv.putText(img, "Roll:"+str(np.round(float(roll_est), 3)), (10,420), font, 1, (0,0,255), 2)
                cv.putText(img, "Pitch:"+str(np.round(float(pitch_est), 3)), (10,440), font, 1, (0,255,0), 2)
                cv.putText(img, "Psi:"+str(np.round(float(yaw_est), 3)), (10,460), font, 1, (255,0,0), 2)


                #Kalman Filter - Position Estimation
                R_IMU = R.from_euler('zyx',[[float(yaw_est), float(pitch_est), float(roll_est)]], degrees=True).as_matrix()
        
                y_pos = np.array([[xf_obj],[yf_obj],[zf_obj]])
                u_pos = R_IMU@np.array([[ax],[ay],[az]]) - np.array([[0],[0],[9.81]])
                # print(u_pos)
                pos_est, P_pos = KF_Position(x_ant_pos, P_ant_pos, u_pos.reshape(3,1), y_pos, dt)
                x_ant_pos = pos_est
                P_ant_pos = P_pos

                x_kf = pos_est[0]
                y_kf = pos_est[1]
                z_kf = pos_est[2]

                cv.putText(img, "X:"+str(np.round(float(x_kf), 4)), (80,700), font, 1, (0,0,0), 2)
                cv.putText(img, "Y:"+str(np.round(float(y_kf), 4)), (180,700), font, 1, (0,0,0), 2)
                cv.putText(img, "Z:"+str(np.round(float(z_kf), 4)), (280,700), font, 1, (0,0,0), 2)


                #Matriz de rotação dinâmica referencia
                if elapsed_time > 10:
                    if gama < np.pi/4:

                        gama += np.pi/800

                    elif beta < np.pi/6:

                        beta += np.pi/800
                    


                R_ref = np.array([[c(alfa)*c(gama) - s(alfa)*c(beta)*s(gama), s(alfa)*c(gama) + c(alfa)*c(beta)*s(gama), s(beta)*s(gama)],
                                [-c(alfa)*s(gama) - s(alfa)*c(beta)*c(gama), -s(alfa)*s(gama)+c(alfa)*c(beta)*c(gama), s(beta)*c(gama)],
                                [s(alfa)*s(beta), -c(alfa)*s(beta), c(beta)]])
                
                R_ref = T_rc[0:3, 0:3]@R_ref

                rvec_ref,_ = cv.Rodrigues(R_ref)

                #Getting quaternions from rotation matrix
                r_refe = sci.spatial.transform.Rotation.from_matrix(T_cr[0:3,0:3]@R_ref)
                q_ref = r_refe.as_quat()
                # # print('Quaternion:')
                # # print(q)

                roll_ref, pitch_ref, yaw_ref = computeAngles(q_ref[3], q_ref[1], q_ref[0], -q_ref[2])

                cv.drawFrameAxes(img, mtx, dist, rvec_ref, np.array([[-0.2],[-0.1],[1.6]]), 0.066, )

                cv.putText(img, "Orientacao Dinamica de Referencia:", (500, 200), font, 1, (255, 255, 255), 2)
                cv.putText(img, "Phi:"+str(np.round(float(roll_ref), 2)), (500,220), font, 1, (0,0,255), 2)
                cv.putText(img, "Theta:"+str(np.round(float(pitch_ref), 2)), (500,240), font, 1, (0,255,0), 2)
                cv.putText(img, "Psi:"+str(np.round(float(yaw_ref), 2)), (500,260), font, 1, (255,0,0), 2)

                #########################################################################################################

                #Write position data in pos_data.txt
                pos_data.write("{:.4f} , ".format(float(xf_obj)) + "{:.4f} , ".format(float(yf_obj)) + "{:.4f} , ".format(float(zf_obj)) + str(time.perf_counter()-start_clock) + "\n")
                #Write attitude data in file
                angle_data.write("{:.2f} , ".format(float(roll_ref)) + "{:.2f} , ".format(float(pitch_ref)) + "{:.2f} , ".format(float(yaw_ref))
                                + "{:.2f} , ".format(float(roll_est)) + "{:.2f} , ".format(float(pitch_est)) + "{:.2f} , ".format(float(yaw_est)) +  str(time.perf_counter()-start_clock) + "\n")
   
                

    
    #Compute FPS
    elapsed_time = time.perf_counter() - start_clock
    print("Start time: {0} \n Elapsed Time: {1}".format(start_clock, elapsed_time))
    fps = int(frame_id / elapsed_time)
    #Print FPS on the screen
    cv.putText(img, "FPS:" + str(fps), (10, 80), font, 1, (255,255,255), 1)
    cv.putText(img, "Time Elapsed:" + str(round(elapsed_time,1)), (100, 80), font, 1, (255,255,255), 1)
    cv.imshow('img', img)
    
    key = cv.waitKey(10)
    if key == ord('n') or key == ord('p'):
        break



pos_data.close()
angle_data.close()
cap.stop()
# cap.release()
cv.destroyAllWindows()



isRun = False
thread.join()
serialConnection.close()


plt.style.use("fivethirtyeight")
fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(10,10), sharex=True)
rr_list, pr_list, yr_list, re_list, pe_list, ye_list, time = [], [], [], [], [], [], []

data = open("angle_data.txt", "r").read()
lines = data.split('\n')

for line in lines:
    if len(line)>1:
        rr, pr, yr, re, pe, ye, times = line.split(' , ')
        rr_list.append(float(rr))
        pr_list.append(float(pr))
        yr_list.append(float(yr))
        re_list.append(float(re))
        pe_list.append(float(pe))
        ye_list.append(float(ye))
        time.append(float(times))


ax.plot(time, rr_list, 'r--', alpha=0.6, label = r'$\phi_{ref} (t)$', linewidth = 1.5)
ax.plot(time, re_list, 'r', label=r'$\phi_{est}(t)$', linewidth=1)
ax2.plot(time, pr_list, 'g--', alpha=0.6, label = r'$\theta_{ref} (t)$', linewidth=1.5)
ax2.plot(time, pe_list, 'g', label=r'$\theta_{est}(t)$', linewidth=1)
ax3.plot(time, yr_list, 'b--', alpha=0.6, label = r'$\psi_{ref} (t)$', linewidth=1.5)
ax3.plot(time, ye_list, 'b', label=r'$\psi_{est}(t)$', linewidth=1)
ax3.set_xlabel('Time (s)')

ax.set_ylabel(r'$\phi$ (°)')
ax2.set_ylabel(r'$\theta$ (°)')
ax3.set_ylabel(r'$\psi$ (°)')
ax.set_title('Orientação')

ax.set_ylim([-90, 90])
ax2.set_ylim([-90, 90])
ax3.set_ylim([-90, 90])
ax.legend()
ax2.legend()
ax3.legend()

plt.tight_layout()
plt.show()


