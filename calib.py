import cv2
import numpy as np
import pickle

#img = cv2.imread('data/checkerboard.jpg')
img = cv2.imread('data/sample-frame.png')

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (7,7), None)
assert ret

# If found, add object points, image points (after refining them)
objpoints.append(objp)

corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
imgpoints.append(corners2)

ret, mtx, dist, rvecs, tvecs = \
    cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
assert ret

pickle.dump({
    'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs
}, open('data/intrinsics.pkl', 'wb'))

# Draw and display the corners
img = cv2.drawChessboardCorners(img, (7,7), corners2, True)
cv2.imshow('frame', img)
cv2.waitKey(30)

import time
time.sleep(5)
