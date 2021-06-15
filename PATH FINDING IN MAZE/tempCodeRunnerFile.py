
# img_path= "C:\\Users\\PRANAV.DESKTOP-9JIUM6F\\OneDrive\\Desktop\\AGV TASK ROUND\\Task 1\\Task_1_Low.png"
# img = cv.imread(img_path, cv.IMREAD_COLOR)

# # defining the colour codes in order B, G, R
# SRC_COLOR = [113, 204, 45]
# DSTN_COLOR = [60, 76, 231]
# OBSTACLE_COLOR = [255 ,255 ,255]
# PATH_COLOR = [180, 0, 90]           # purple color to show the final path obtained
# TRAVERSED_COLOR = [0, 150, 250]    # orange to represent nodes which are in closed list
# CURRENT_COLOR = [250, 200, 0]    # blue to represent those nodes which are currently in open-list

# # finding the coordinates of the source and destination:
# def src_dstn_find(image, h, w):
#     Flag1 = False
#     Flag2 = False
#     for i in range(h):
#         for j in range(w):
#             if (img[i][j] == SRC_COLOR).all():
#                 src = (i,j)
#                 Flag1 = True
#             if (img[i][j] == DSTN_COLOR).all():
#                 dstn = (i,j)
#                 Flag2 = True 
#     if (Flag1 == False):
#         print('The source pixel could not be found in the image.')
#     if (Flag2 == False):
#         print('The destination piel could not be found in the image.')
#     if (Flag1 == True and Flag2 == True):
#         print(f"Coordinates of the source :{src}")
#         print(f"Coordinates of the destination:{dstn}")

#     return (src, dstn)