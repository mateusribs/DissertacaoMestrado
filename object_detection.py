from cv2 import cv2
import numpy as np
from imutils.video import WebcamVideoStream
import time
import math 

#Classe Camera
cap = WebcamVideoStream(src=0).start()
#cap = cv2.VideoCapture("Ball1.mp4")
#Calculo de FPS
start_time = time.time()
#Setup de fonte
font = cv2.FONT_HERSHEY_PLAIN

cX = None
cY = None
frame_id = 0
#Loop de captura de frames
while True:
    #Carrega frame
    frame = cap.read()
    
    frame_id += 1
    #Converte frame para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #Detecção de cor através de HSV
    lower = np.array([118, 89, 19])
    upper = np.array([179, 255, 255])
    #Cria mascara para filtrar o objeto pela cor definida pelos limites
    mask = cv2.inRange(hsv, lower, upper)
    #Cria kernel
    kernel = np.ones((5,5), np.uint8)
    #Aplica processo de Abertura (Erosão seguido de Dilatação)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 4)
    
    
    _, cnts, _ = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    
    
    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        perimeter = cv2.arcLength(c, True)
        metric = (4*math.pi*M["m00"])/perimeter**2
        
        if metric > 0.8:
        
            #draw the contour and center of the shape on the image
            cv2.drawContours(frame, [c], -1, (255, 0, 0), 2)
            cv2.circle(frame, (cX, cY), 1, (255, 0, 0), 2)
    
    
    elapsed_time = time.time() - start_time
    fps = int(frame_id / elapsed_time)
    cv2.putText(frame,"FPS:" + str(fps) + " Center:"+str(cX)+
    ','+str(cY), (10, 80), font, 1, (255,255,255), 1)
    cv2.imshow("Original Image", frame)
    cv2.imshow("Processed Image", opening)
    
    key = cv2.waitKey(10)
    if key == 27:
        break
  
cap.stop()
cv2.destroyAllWindows()