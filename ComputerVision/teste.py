import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)

if not cap.isOpened():
    print("Error opening file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame',frame)
    key = cv2.waitKey(10)
    if key == ord('n') or key == ord('p'):
        break