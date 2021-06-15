
import cv2 as cv
import numpy as np
import os
import glob

# DIMENSIONS OF CHECKERBOARD
CHECKERBOARD = (6, 8)        # checkerboard size is as per inner corners (not including the corners formed by outer edges)     
SIZE = CHECKERBOARD[0]*CHECKERBOARD[1]


# TERMINATION CRITERION (to stop the iteration when the desired accuracy is reached or specified number of iterations done)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points (3d point in real world space) and image points (2d points in image plane) from all images.
objpoints = []              # list containing 3_D  world coordinates of corners in all images       
imgpoints = []              #  list containing 2_D  image coordinates of corners in all images

# defining the coordinates for the chessboard corners in its own frame
threeD_pts = np.zeros((SIZE, 3), np.float32)
threeD_pts[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Method 2 for defining the chessboard corners in its own frame (my own implementation):
# threeD_pts = np.zeros((SIZE, 3), np.float32)
# for i in range(0, SIZE):  
#     threeD_pts[i][0] = i%(CHECKERBOARD[0])
#     threeD_pts[i][1] = int(i/CHECKERBOARD[0])
# print(threeD_pts)


#glob function is used to extract the path of individual files present within same directory
images = glob.glob(r'Task 5_20CS30037\Calib_Imgs\*.jpg')
# for filename in images:
#     print(filename)

# count variable is defined to count the number of images in which corners have been detected successfully.
count = 0

for filename in images:
    image = cv.imread(filename, 1)
    print(image.shape)
    grayImg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Using the openCV function "findChessboardCorners" to find the corners of the checkerboard
    # If ret value is True, it means desired no of corners have been detected
    ret, corners = cv.findChessboardCorners(grayImg, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK +  cv.CALIB_CB_NORMALIZE_IMAGE)
    
    # If desired number of corners detected (ret = True) then, refine the pixel coordinates to higher accuracy using 'cornerSubPix'
    if ret == True:
        objpoints.append(threeD_pts)
        count = count+1
    
        corners2 = cv.cornerSubPix(grayImg, corners, (11, 11), (-1, -1), criteria)  
        imgpoints.append(corners2)

        # drawChessboardCorners returns the image with the corners marked as red circles or as colored corners connected by lines
        image = cv.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
    else:
        print("The chessboard corners in the given image could not be detected properly.")
    
    cv.namedWindow('Corner_detect', cv.WINDOW_NORMAL)
    cv.imshow('Corner_detect', image)
    k = cv.waitKey(0)
    if (k == ord('q')):
        cv.destroyAllWindows()

print(count)
  
h, w = image.shape[:2]
    
# Camera calibration by passing the value of above found out 3D points(objpoints) and corresponding pixel coordinates of the detected corners(imgpoints)
ret, Cam_matrix, distortion, r_vecs, t_vecs = cv.calibrateCamera(objpoints, imgpoints, grayImg.shape[::-1], None, None)
file1 = open(r"Task 5_20CS30037\Calib_res.txt", "w")

# Output
print("Camera matrix:")
print(Cam_matrix)
  
print("\nDistortion coefficient matrix:")
print(distortion)
  
print("\nRotation Vectors:")
print(r_vecs)
  
print("\nTranslation Vectors:")
print(t_vecs)

# Saving the Camera_Calibration data (intrinsic matrix and destortion matrix) and the roation and translation vectors to a text file "Calib_res"
L = [f"Camera matrix:\n{Cam_matrix}\n\nDistortion coefficient matrix:\n{distortion}\n\nRotation Vectors:\n{r_vecs}\n\nTranslation Vectors:\n{t_vecs}"]
file1.writelines(L)
file1.close()

# Saving the Camera_matrix and Distortion_matrix to a xml file "Calibration_result" for using in aruco marker pose estimation:
path = r"Task 5_20CS30037\Calibration_result.xml"
file2 = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
file2.write("Camera_matrix", Cam_matrix)
file2.write("Distortion_matrix", distortion)
file2.release()






