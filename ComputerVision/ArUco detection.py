import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
from imutils.video import WebcamVideoStream
import matplotlib.pyplot as plt
import glob
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

#Camera instance with thread
# cap = WebcamVideoStream(src=0).start()
cap = cv.VideoCapture('teste_2.mp4')

#Create board object
board = cv.aruco.GridBoard_create(2, 2, 0.028, 0.003, dictionary)

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

while True:
    ret, img = cap.read()

    if ret:
        #Convert frame to gray scale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #Frame count increment
        frame_id += 1

        #Set parameters for the marker tracking
        parameters = cv.aruco.DetectorParameters_create()

        #Detect the markers in the image
        markerCorners, markerIDs, rejectedCandidates = cv.aruco.detectMarkers(gray, dictionary, parameters = parameters)
        #Refine detection
        markerCorners, markerIDs, rejectedCandidates, recoveredIds = cv.aruco.refineDetectedMarkers(img, board, markerCorners,
                                                                                                    markerIDs, rejectedCandidates, mtx, dist)
        #Open position file to write data position
        pos_data = open("pos_data.txt", "a+")

        #Verify if there is some marker detected
        if markerIDs is not None and len(markerIDs) > 0:
            # print("Marker Detected")
            #Compute board's pose
            pose, rvecs, tvecs = cv.aruco.estimatePoseBoard(markerCorners, markerIDs, board, mtx, dist, rvecs, tvecs)
            #Use Rodrigues formula to transform rotation vector into matrix
            R_matrix, _ = cv.Rodrigues(rvecs)

            if pose:
                cv.aruco.drawAxis(img, mtx, dist, rvecs, tvecs, 0.028)

            cv.aruco.drawDetectedMarkers(img, markerCorners)   
            
            #Getting quaternions from rotation matrix
            r = R.from_matrix(R_matrix)
            q = r.as_quat()

            #Calibration position: In the beggining of the video, the script will get 40 early datas to compute the mean
            #measurements and so find offset values. Having the offset, we can calculate the relative origin.
            #Keep in mind that the marker's position was acquired with respect to camera frame.
            if calib_pos:

                pos_x.append(float(tvecs[0]))
                pos_y.append(float(tvecs[1]))
                pos_z.append(float(tvecs[2]))

                if len(pos_x) == 40 and len(pos_y) == 40 and len(pos_z):
                    
                    offset_x = sum(pos_x)/len(pos_x)
                    offset_y = sum(pos_y)/len(pos_y)
                    offset_z = sum(pos_z)/len(pos_z)
                    
                    # print(offset_x)
                    # print(offset_y)
                    # print(offset_z)

                    calib_pos = False

            #Marker's position
            xf = float(tvecs[0]) - offset_x
            yf = float(tvecs[1]) - offset_y
            zf = float(tvecs[2]) - offset_z

            # print('Rotation Matrix:', R_matrix)
            # print("Quaternion: ", q)

            #Attitude estimation
            # phi_est = 180*math.atan2(R_matrix[2,1], R_matrix[2,2])/math.pi
            # theta_est = 180*math.atan2(-R_matrix[2, 0], math.sqrt(R_matrix[2,1]**2 + R_matrix[2,2]**2))/math.pi
            # psi_est = 180*math.atan2(R_matrix[1,0], R_matrix[0,0])/math.pi
            # print("Estimated Angles")
            # print("Roll:",phi_est," Pitch:",theta_est," Yaw:",psi_est)
            # # print("Matriz Rotação Original:", quad_rot)
            # print("Matriz Rotação Estimada:", R_matrix)   
            
            #Write position data in pos_data.txt
            pos_data.write("{:.4f} , ".format(xf) + "{:.4f} , ".format(yf) + "{:.4f} , ".format(zf) + str(time.process_time()-start_clock) + "\n")

            # last_col = np.array([[0, 0, 0, 1]])
            # T_mat = np.concatenate((R_matrix, tvecs), axis=1)
            # T_mat = np.concatenate((T_mat, last_col), axis=0)

            #Print position values in frame
            cv.putText(img, "X:"+str(np.round(xf, 4)), (50,700), font, 1, (255,255,255), 2)
            cv.putText(img, "Y:"+str(np.round(yf, 4)), (200,700), font, 1, (255,255,255), 2)
            cv.putText(img, "Z:"+str(np.round(zf, 4)), (350,700), font, 1, (255,255,255), 2)
            
            
            
            

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
# cap.stop()
cv.destroyAllWindows()


plt.style.use("fivethirtyeight")
fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(10,10), sharex=True)
x, y, z, time = [], [], [], []

data = open("pos_data.txt", "r").read()
lines = data.split('\n')

for line in lines:
    if len(line)>1:
        xs, ys, zs, times = line.split(' , ')
        x.append(float(xs))
        y.append(float(ys))
        z.append(float(zs))
        time.append(float(times))


ax.plot(time, x, 'r', label = r'$x(t)$', linewidth = 1)
ax2.plot(time, y, 'g', label = r'$y(t)$', linewidth=1)
ax3.plot(time, z, 'b', label = r'$z(t)$', linewidth=1)
ax3.set_xlabel('Time (s)')

ax.set_ylabel(r'$x$ (meters)')
ax2.set_ylabel(r'$y$ (meters)')
ax3.set_ylabel(r'$z$ (meters)')
ax.set_title('Posição')

ax.set_ylim([-0.1, 0.1])
ax2.set_ylim([-0.1, 0.1])
ax3.set_ylim([-0.1, 0.1])
ax.legend()
ax2.legend()
ax3.legend()

plt.tight_layout()
plt.show()


