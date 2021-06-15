import cv2 as cv
import numpy as np
from cv2 import aruco
import math

# Importing the camera matrix and distortion matrices obtained from Camera Calibration using the xml file:
path = r"Task 5_20CS30037\Calibration_result.xml"
file1 = cv.FileStorage(path, cv.FILE_STORAGE_READ)
Camera_matrix = file1.getNode("Camera_matrix").mat()
Distortion_matrix = file1.getNode("Distortion_matrix").mat()
file1.release()

# print(Camera_matrix)
# print(Distortion_matrix)

# # # # # # # # ARUCO MARKER GENERATION USIMG ARUCO LIBRARY # # # # # # # #

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
marker_img = np.zeros((200, 200), dtype = np.uint8)
marker_img = aruco.drawMarker(aruco_dict, 12, 500, marker_img, 1)
cv.imwrite(r"Task 5_20CS30037\ArUco_marker_img.png", marker_img)
cv.namedWindow("Aruco Marker", cv.WINDOW_NORMAL)
cv.imshow("Aruco Marker", marker_img)
k = cv.waitKey(10000)
if (k == ord('q')):
    cv.destroyAllWindows()

markerLength = 13.2  #   length in  centimeters 
# # # # # # # # # # # # ARUCO MARKER DETECTION AND POSE ESTIMATION # # # # # # # # # 


# Initialising marker detection parameters using the defaul parameter values (the parameters we are going to use for ArUco detection):
ArUcoParam = aruco.DetectorParameters_create()


webcam = cv.VideoCapture(0)
if not webcam.isOpened(): 
    print("Error while opening the camera.")
    exit(-1)

frame = np.array([])

while True:

    # Capturing the input video frame - by - frame
    ret, frame = webcam.read()
    print(frame.shape)
  
    # Converting colored image to grayscale image 
    grayImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Aruco detection:
    markerCorners, markerIds, rejectedImgPoints= aruco.detectMarkers(grayImg, aruco_dict, parameters = ArUcoParam, cameraMatrix = Camera_matrix, distCoeff = Distortion_matrix)

    if (len(markerCorners) > 0): # to ensure atleast one Aruco tag was detected
        
        # converting the array storing ids to a 1-D array
        markerIds = markerIds.flatten()
        
        # drawing  a square boundary around each detected marker
        aruco.drawDetectedMarkers(frame, markerCorners, markerIds, borderColor = (0, 255, 0)) 

        # Iterating through each marker to obtain its corner coordinates
        for i in range(0, len(markerIds)):            
            
            corners = markerCorners[i].reshape((4, 2))
            (c1, c2, c3, c4) = corners
            # convert each of the (x, y)-coordinate pairs for the corner pixels to integers (as default value type of numpy array elements is float, although the value stored are integers)
            c1 = (int(c1[0]), int(c1[1]))  # Top-left corner
            c2 = (int(c2[0]), int(c2[1]))  # Top-right corner
            c3 = (int(c3[0]), int(c3[1]))  # Bottom-left corner
            c4 = (int(c4[0]), int(c4[1]))  # Bottom-right corner

            # Pose estimation function
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(markerCorners[i], markerLength, Camera_matrix, Distortion_matrix)
            # markerPoints is the array of object points of all the marker corners
            tvec = tvec.reshape(-1, 3)

            # to draw the axes of the world coordinate system
            aruco.drawAxis(frame, Camera_matrix, Distortion_matrix, rvec, tvec, 15)

            #  to calculate the distance of the Aruco marker from the Camera centre and disply it on the screen:
            dist = math.sqrt(tvec[0, 0]**2 + tvec[0, 1]**2 + tvec[0, 2]**2)
            cv.putText(frame, text = f"X = {int(tvec[0,0])} cms", org = (20, 50), fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255, 0, 0), thickness = 2,lineType = cv.LINE_AA)
            cv.putText(frame, text = f"Y (Height) = {int(tvec[0,1])} cms", org = (20, 80), fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0, 255, 0), thickness = 2,lineType = cv.LINE_AA)
            cv.putText(frame, text = f"Z = {int(tvec[0,2])} cms", org = (20, 110), fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0, 0, 255), thickness = 2,lineType = cv.LINE_AA)
            cv.putText(frame, text = f"Distance = {int(dist)} cms", org = (20, 140), fontFace = cv.FONT_HERSHEY_SIMPLEX , fontScale = 0.5, color = (255, 255, 0), thickness = 2 ,lineType = cv.LINE_AA)


            # Using the Rodrigues function to convert our rotation vector to a rotation matrix:
            rot_mat = np.zeros((3, 3), dtype = float)
            rot_mat, _ = cv.Rodrigues(rvec)
            
            # rot_mat = [ r11, r12, r13
            #             r21, r22, r23
            #             r31, r32, r33 ]

            r11 = rot_mat[0, 0]
            r12 = rot_mat[0, 1]
            r13 = rot_mat[0, 2]
            r21 = rot_mat[1, 0]
            r22 = rot_mat[1, 1]
            r23 = rot_mat[1, 2]
            r31 = rot_mat[2, 0]
            r32 = rot_mat[2, 1]
            r33 = rot_mat[2, 2]

            # finding roll, pitch, yaw from rot_mat and displaying on the output screen

            yaw = math.degrees(math.atan(r21/r11))                                        # rotation wrt Z axis
            pitch = math.degrees(math.atan(-r31/(math.sqrt(r32**2 + r33**2))))            # rotation wrt Y axis
            roll = math.degrees(math.atan(r32/r33))                                       # rotation wrt X axis

            cv.putText(frame, text = f"Yaw(Z) = {int(yaw)}" , org = (500, 50), fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255, 0, 0), thickness = 2, lineType = cv.LINE_AA)
            cv.putText(frame, text = f"Pitch(Y) = {int(pitch)}" ,org =  (500,80), fontFace =  cv.FONT_HERSHEY_SIMPLEX, fontScale =  0.5, color =  (0, 255, 0), thickness =  2, lineType = cv.LINE_AA)
            cv.putText(frame, text = f"Roll(X) = {int(roll)}" ,org =  (500,110), fontFace =  cv.FONT_HERSHEY_SIMPLEX, fontScale =  0.5, color =  (0, 0, 255), thickness =  2, lineType = cv.LINE_AA)
     
    # Displaying the resulting frame
        
        cv.namedWindow("Output Screen", cv.WINDOW_AUTOSIZE)
        cv.imshow("Output Screen", frame)
        key = cv.waitKey(50) & 0xFF
        if (key == ord('q')):  
            webcam.release()
            cv.destroyAllWindows()
