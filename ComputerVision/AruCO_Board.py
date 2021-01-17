import numpy as np
import cv2 as cv

dictionar = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
board = cv.aruco.GridBoard_create(2, 2, 0.05, 0.005, dictionar)

img = board.draw((680,500), 10, 1)

cv.imwrite('aruco_board.png', img)

# Display the image to us
cv.imshow('Gridboard', img)
# Exit on any key
cv.waitKey(0)
cv.destroyAllWindows()