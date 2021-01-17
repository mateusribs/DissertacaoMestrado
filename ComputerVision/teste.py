import cv2

cap = cv2.VideoCapture('teste.MP4')

if not cap.isOpened():
    print("Error opening file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame',frame)
    key = cv2.waitKey(10)
    if key == ord('n') or key == ord('p'):
        break