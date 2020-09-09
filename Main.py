import numpy as np
import cv2 as cv
from imutils.video import WebcamVideoStream
import glob
import time
import math 

# Load previously saved calibration data
path = './camera_data/camera_calibration.npz'
npzfile = np.load(path)
#Camera Matrix
mtx = npzfile[npzfile.files[0]]
#Distortion Matrix
dist = npzfile[npzfile.files[1]]
rvec = npzfile[npzfile.files[2]]
tve = npzfile[npzfile.files[3]]
print(mtx ,dist)

#Font setup
font = cv.FONT_HERSHEY_PLAIN
start_time = time.time()

objpoints = np.array([[0, 0, 0], [0.0255, 0, 0], [-0.0255, 0, 0],[0, 0.0255, 0], [0, -0.0255, 0],
                      [0, 0.0755, 0], [0.0255, 0.0755, 0], [-0.0255, 0.0755, 0], [0, 0.05, 0], [0, 0.101, 0],
                      [0.1055, 0, 0], [0.131, 0, 0], [0.08, 0, 0], [0.1055, 0.0255, 0], [0.1055, -0.0255, 0]], dtype=np.float32)

def nothing(x):
    pass

#Criar janela para trackbar
cv.namedWindow("Trackbars")

#Criar trackbars
cv.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U - V", "Trackbars", 255, 255, nothing)


def detect_contourn(image, color):
        
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    if color == "Red":
        #Define the limits in HSV variables
        lower = np.array([0, 127, 62])
        upper = np.array([20, 255, 255])
    if color == "Green":
        #Define the limits in HSV variables
        lower = np.array([30, 44, 67])
        upper = np.array([91, 255, 255])
    if color == "Blue":
        #Define the limits in HSV variables
        lower = np.array([65, 107, 86])
        upper = np.array([148, 236, 255])
    if color == "Yellow":
        #Define the limits in HSV variables
        lower = np.array([20, 100, 100])
        upper = np.array([32, 220, 255])

    l_h = cv.getTrackbarPos("L - H", "Trackbars")
    l_s = cv.getTrackbarPos("L - S", "Trackbars")
    l_v = cv.getTrackbarPos("L - V", "Trackbars")
    u_h = cv.getTrackbarPos("U - H", "Trackbars")
    u_s = cv.getTrackbarPos("U - S", "Trackbars")
    u_v = cv.getTrackbarPos("U - V", "Trackbars")  

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])

    #Define threshold for red color
    mask = cv.inRange(hsv, lower, upper)
    # cv.imshow("Mask", mask)
    #Create a kernel
    kernel = np.ones((5,5), np.uint8)
    #Apply opening process
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations = 5)
    output = cv.bitwise_and(image, image, mask=opening)
    cv.imshow("Opening", output)
    #Find BLOB's contours
    cnts, _ = cv.findContours(opening.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    return  cnts

def center_mass_calculate(image, c):

    # Compute the center of the contour
    M = cv.moments(c)
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])

    (cX, cY), radius = cv.minEnclosingCircle(c)
    cX = int(cX)
    cY = int(cY)
    center = (cX, cY)
    radius = int(radius)
    perimeter = cv.arcLength(c, True)
    #Compute the eccentricity
    metric = (4*math.pi*M["m00"])/perimeter**2
    if metric > 0.8:
        #Draw the contour and center of the shape on the image
        # cv.drawContours(image, [c], -1, (0, 0, 0), 1)
        cv.circle(image, center, radius, (0, 0, 0), 1)
        cv.circle(image, center, 1, (0, 0, 0), -1)

                          
    return cX, cY, radius

def draw(image, imgpoints, imgpts):
    imgpoint = tuple(imgpoints[0].ravel())
    img = cv.line(image, imgpoint, tuple(imgpts[0].ravel()), (255,0,0), 3)
    img = cv.line(image, imgpoint, tuple(imgpts[1].ravel()), (0,255,0), 3)
    img = cv.line(image, imgpoint, tuple(imgpts[2].ravel()), (0,255,255), 3)
    return img
    
def get_element_vector(f1, f2, c1, c2):
    #Where f1 is frameA, f2 is frameB
    #c1 is the coordinate, let x = 0, y = 1, z = 2
    vec = []
    for i in range(np.shape(f1)[0]):
        cc = f2[i, c1]*f1[i, c2]
        vec.append(cc)
    return np.sum(vec)

def get_element_A(f1, c1, c2):
    A = []
    for i in range(np.shape(f1)[0]):
        cc = f1[i, c1]*f1[i, c2]
        A.append(cc)
    return np.sum(A)

def get_element_last(f1, c1):
    last = []
    for i in range(np.shape(f1)[0]):
        cc = f1[i, c1]
        last.append(cc)
    return np.sum(last)

def get_transform_frame(f1, f2):
    matrix = np.zeros((3,4))
    for i in range(3):
        for j in range(3):
            matrix[i, j] = get_element_vector(f1, f2, i, j)
            matrix[i, 3] = get_element_last(f2, i)

    A = np.zeros((4,4))
    for i in range(3):
        for j in range(3):
            A[i, j] = get_element_A(f1, i, j)

    for i in range(3):
        A[i,3] = get_element_last(f1, i)
        A[3, i] = get_element_last(f1, i)

    A[3,3] = np.shape(f1)[0]
    A_inv = np.linalg.inv(A)

    matrix = np.transpose(matrix)

    T = np.dot(A_inv, matrix)
    T = np.transpose(T)
    last_row = np.array([0,0,0,1]).reshape(1,4)
    T = np.concatenate((T, last_row), axis=0)
        
    return T

def get_pose(image, objpoints, imgpoints, mtx, dist):
        
    axis = np.float32([[.05, 0, 0], [0, .05, 0], [0, 0, .05]]).reshape(-1,3)

    # Find the rotation and translation vectors.
    _, rvecs, tvecs, _ = cv.solvePnPRansac(objpoints, imgpoints, mtx, dist, iterationsCount=5)
    # project 3D points to image plane
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
    imgpts = imgpts.astype(np.int)
    img = draw(image, imgpoints, imgpts)
    R_matrix, _ = cv.Rodrigues(rvecs)
    
    return rvecs, tvecs, R_matrix, image


#Camera instance with thread
cap = WebcamVideoStream(src=0).start()

frame_id = 0

#Image constants
cX1 = 0
cY1 = 0
r1 = 0
cX2 = 0
cY2 = 0
r2 = 0
cX3 = 0
cY3 = 0
r3 = 0

#Transformation frame constants
obj_frame = []
ground_frame = []
distances = []
T_flag = False


while True:
    img = cap.read()
    frame_id += 1

    #Get the circle's contourn based on it color
    cnts_red = detect_contourn(img, "Red")
    cnts_green = detect_contourn(img, "Green")
    cnts_blue = detect_contourn(img, "Blue")

    #Get the center point of the detected circle
    for c in cnts_red:
        cX1, cY1, r1 = center_mass_calculate(img, c)
    for c in cnts_green:
        cX2, cY2, r2 = center_mass_calculate(img, c)
    for c in cnts_blue:
        cX3, cY3, r3 = center_mass_calculate(img, c)

    
    #Set the image points
    imgpoints = np.array([[cX1, cY1], [cX1+r1, cY1], [cX1-r1, cY1], [cX1, cY1+r1], [cX1, cY1-r1],
                         [cX2, cY2], [cX2+r2, cY2], [cX2-r2, cY2], [cX2, cY2+r2], [cX2, cY2-r2],
                         [cX3, cY3], [cX3+r3, cY3], [cX3-r3, cY3], [cX3, cY3+r3], [cX3, cY3-r3]], dtype = np.float32)

    # Set Frame Transformation parameters
    if cX1 != 0 and cY1 != 0:
        global R_matrix

        #Get the extrinsics parameters
        rvecs, tvecs, R_matrix, image = get_pose(img, objpoints, imgpoints, mtx, dist)
        tvec = np.concatenate((tvecs, np.ones((1,1))), axis=0)
        print("Tvec:", tvec)

        #List containing object frame points
        obj_pos = np.reshape(tvecs, (1,3))
        obj_pos = np.asarray(obj_pos, np.float32)
        obj_frame.append(obj_pos)

        #List containing ground frame points (SENSOR)
        ground_pos = np.zeros((1,3))
        ground_pos = np.asarray(ground_pos, np.float32)
        ground_frame.append(ground_pos)

        # #Euler angles from quadrotor (SENSOR - IMU)
        # quad_rot = quad_position.env.mat_rot
        # phi_quad = 180*math.atan2(-quad_rot[2,1], quad_rot[2,2])/math.pi
        # theta_quad = 180*math.asin(quad_rot[2,0])/math.pi
        # psi_quad = 180*math.atan2(-quad_rot[1,0], [quad_rot[0,0]])/math.pi
        # print("Quadrotor Angles")
        # print("Roll:",phi_quad," Pitch:",theta_quad," Yaw:",psi_quad)
        #Estimated angles from camera
        phi_est = 180*math.atan2(-R_matrix[2,1], R_matrix[2,2])/math.pi
        theta_est = 180*math.asin(R_matrix[2, 0])/math.pi
        psi_est = 180*math.atan2(-R_matrix[1,0], R_matrix[0,0])/math.pi
        print("Estimated Angles")
        print("Roll:",phi_est," Pitch:",theta_est," Yaw:",psi_est)
        # print("Matriz Rotação Original:", quad_rot)
        print("Matriz Rotação Estimada:", R_matrix)
        cv.putText(img, "Roll:"+str(round(phi_est, 2)), (50,440), font, 1, (255,255,255), 2)
        cv.putText(img, "Pitch:"+str(round(theta_est, 2)), (200,440), font, 1, (255,255,255), 2)
        cv.putText(img, "Yaw:"+str(round(psi_est, 2)), (350,440), font, 1, (255,255,255), 2)

    #Evaluate transformation matrix beetween object frame and ground frame
    if len(obj_frame)==10 and len(ground_frame)==10:
        global T

        obj_frame = np.asarray(obj_frame, np.float32).reshape(10,3)
        ground_frame = np.asarray(ground_frame, np.float32).reshape(10,3)

        T = get_transform_frame(obj_frame, ground_frame)
        T_flag = True
        obj_frame = []
        ground_frame = []

   

        if T_flag:
            real_pos = np.dot(T, tvec)
            # erro_X = quad_position.env.state[0] - real_pos[0]
            # erro_Y = quad_position.env.state[2] - real_pos[1]
            # erro_Z = quad_position.env.state[4] - real_pos[2]
            print("Ground Frame:", ground_pos)
            print("Real Frame:", np.transpose(real_pos)[:,:3])
            # print("E_X:",erro_X," E_Y:",erro_Y," E_Y:", erro_Z)
            T_flag = False


    # Print the image coordinates on the screen
    cv.putText(img," Center:"+str(cX1)+','+str(cY1), (10, 10), font, 1, (255,0,0), 1)
    cv.putText(img," Center:"+str(cX2)+','+str(cY2), (10, 25), font, 1, (0,255,0), 1)
    cv.putText(img," Center:"+str(cX3)+','+str(cY3), (10, 40), font, 1, (0,0,255), 1)
    # cv.putText(image," Center:"+str(cX4)+','+str(cY4), (10, 55), font, 1, (0,255,255), 1)

    #Compute FPS
    elapsed_time = time.time() - start_time
    fps = int(frame_id / elapsed_time)
    #Print FPS on the screen
    cv.putText(img, "FPS:" + str(fps), (10, 80), font, 1, (255,255,255), 1)
    
    cv.imshow('img',img)
        
    key = cv.waitKey(10)
    if key == 27:
        break

cap.stop()
cv.destroyAllWindows()