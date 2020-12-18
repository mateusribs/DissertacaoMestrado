import cv2 as cv
import numpy as np
from imutils.video import WebcamVideoStream
import matplotlib.pyplot as plt
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
rvec = npzfile[npzfile.files[2]]
tvec = npzfile[npzfile.files[3]]
print(mtx ,dist)

#Font setup
font = cv.FONT_HERSHEY_PLAIN
start_time = time.time()

# #Generate the marker
# markerImage = np.zeros((200,200), dtype=np.uint8)
# markerImage = cv.aruco.drawMarker(dictionary, 39, 200, markerImage, 1)

# cv.imwrite("marker39.png", markerImage)

angle_data = open("angle_data.txt",'w')
angle_data.close()

#Camera instance with thread
cap = WebcamVideoStream(src=0).start()

start_clock = time.clock()

frame_id = 0
while True:
    img = cap.read()
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    frame_id += 1

    parameters = cv.aruco.DetectorParameters_create()

    #Detect the markers in the image
    markerCorners, markerIDs, rejectedCandidates = cv.aruco.detectMarkers(img, dictionary, parameters = parameters)

   

    angle_data = open("angle_data.txt", "a+")

    if np.all(markerIDs != None):
        rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(markerCorners, 5.3, mtx, dist)
        R_matrix, _ = cv.Rodrigues(rvecs)

        print("tvecs:", tvecs)

        for i in range(0, markerIDs.size):
            cv.aruco.drawAxis(img, mtx, dist, rvecs[i], tvecs[i], 5)

        cv.aruco.drawDetectedMarkers(img, markerCorners)    

    
        phi_est = 180*math.atan2(R_matrix[2,1], R_matrix[2,2])/math.pi
        theta_est = 180*math.atan2(-R_matrix[2, 0], math.sqrt(R_matrix[2,1]**2 + R_matrix[2,2]**2))/math.pi
        psi_est = 180*math.atan2(R_matrix[1,0], R_matrix[0,0])/math.pi
        # print("Estimated Angles")
        # print("Roll:",phi_est," Pitch:",theta_est," Yaw:",psi_est)
        # # print("Matriz Rotação Original:", quad_rot)
        # print("Matriz Rotação Estimada:", R_matrix)
        cv.putText(img, "Roll:"+str(round(phi_est, 2)), (50,440), font, 1, (255,255,255), 2)
        cv.putText(img, "Pitch:"+str(round(theta_est, 2)), (200,440), font, 1, (255,255,255), 2)
        cv.putText(img, "Yaw:"+str(round(psi_est, 2)), (350,440), font, 1, (255,255,255), 2)

        
        
        angle_data.write("{:.4f} , ".format(phi_est) + "{:.4f} , ".format(theta_est) + "{:.4f} , ".format(psi_est) + str(time.clock()-start_clock) + "\n")
    
        
        
        
        

    #Compute FPS
    elapsed_time = time.time() - start_time
    fps = int(frame_id / elapsed_time)
    #Print FPS on the screen
    cv.putText(img, "FPS:" + str(fps), (10, 80), font, 1, (255,255,255), 1)

    cv.imshow('img', img)
        
    key = cv.waitKey(10)
    if key == 27:
        break

angle_data.close()
cap.stop()
cv.destroyAllWindows()


plt.style.use("fivethirtyeight")
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
phi, theta, psi, time = [], [], [], []

data = open("angle_data.txt", "r").read()
lines = data.split('\n')

for line in lines:
    if len(line)>1:
        phis, thetas, psis, times = line.split(' , ')
        phi.append(float(phis))
        theta.append(float(thetas))
        psi.append(float(psis))
        time.append(float(times))


ax.plot(time, phi, label = 'phi (roll)', linewidth = 1)
ax.plot(time, theta, label = 'theta (pitch)', linewidth=1)
ax.plot(time, psi, label = 'psi (yaw)', linewidth=1)
plt.ylim(-180, 180)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (degrees)')
ax.set_title('Estimated Orientation')
ax.legend()


plt.tight_layout()
plt.show()

