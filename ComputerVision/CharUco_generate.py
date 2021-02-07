import cv2 as cv
import numpy as np
# Create ChArUco board, which is a set of Aruco markers in a chessboard setting
# meant for calibration
# the following call gets a ChArUco board of tiles 5 wide X 7 tall
# gridboard = cv.aruco.CharucoBoard_create(
#         squaresX=6, 
#         squaresY=8, 
#         squareLength=0.04, 
#         markerLength=0.02, 
#         dictionary=cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50))

# dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
# gridboard = cv.aruco.GridBoard_create(2, 2, 0.1, 0.3, dictionary)

# Create an image from the gridboard
# img = np.zeros((400,400))
# img = gridboard.draw((400, 400), img, 10, 1)
img = np.zeros((250, 250, 1), dtype='uint8')
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36H11)
cv.aruco.drawMarker(dictionary, 4, 250, img, 1)

cv.imwrite("aruco_marker.jpg", img)

# Display the image to us
cv.imshow('Gridboard', img)
# Exit on any key
cv.waitKey(0)
cv.destroyAllWindows()