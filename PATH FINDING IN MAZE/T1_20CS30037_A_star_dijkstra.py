import numpy as np
import cv2 as cv
import math
import time
import heapq

# key ==> a bool value , if FALSE => NON_DIAGONAL CASE, if TRUE => DIAGONAL CASE
key  = True

# defining the colour codes in order B, G, R
SRC_COLOR = [113, 204, 45]
DSTN_COLOR = [60, 76, 231]
OBSTACLE_COLOR = [255 ,255 ,255]
PATH_COLOR = [180, 0, 90]                     # purple color to show the final path obtained
VISITED_COLOR = [0, 150, 250]                 # orange to represent nodes which are in closed category
CURRENT_COLOR = [250, 200, 0]                 # blue to represent those nodes which are currently in open-list

# function to find the coordinates of the source and destination:
def src_dstn_find(image, h, w):
    Flag1 = False
    Flag2 = False
    for i in range(h):
        for j in range(w):
            if (image[i][j] == SRC_COLOR).all():
                src = (i,j)
                Flag1 = True
            if (image[i][j] == DSTN_COLOR).all():
                dstn = (i,j)
                Flag2 = True 
    if (Flag1 == False):
        print('The source pixel could not be found in the image.')
    if (Flag2 == False):
        print('The destination pixel could not be found in the image.')
    if (Flag1 == True and Flag2 == True):
        print(f"Coordinates of the source :{src}")
        print(f"Coordinates of the destination:{dstn}")

    return (src, dstn)

# defining a class for storing info about each node:
class Node():

    def __init__(self, position, parent):

        self.position = position
        self.parent = parent
        self.g = np.inf
        self.h = np.inf
        self.f = np.inf
        self.closed = False                     
        self.open = False

    def __eq__(self, other):                       # to define condition for equality of two nodes
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __gt__(self, other):
        return self.f > other.f

    def __repr__(self):                            # represent function defined to print info regarding Node type objects
        return (f" {self.position}\ng = {self.g}\th = {self.h}\tf = {self.f}\tparent node = {self.parent}")

# Function to check whether a pixel is a valid pixel or not
def isValid(img: np.ndarray = None, p: tuple = None):
    x, y = p
    if (x>= 0  and y>= 0 and x < img.shape[0] and y < img.shape[1] and (img[x, y] != OBSTACLE_COLOR).any()):
        return True
    else:
        return False

# Function to calculate Heuristic values:
def heuristic(p1: tuple = None, p2: tuple = None):
    x1, y1 = p1
    x2, y2 = p2
    h1 = (abs(x1 - x2) + abs(y1 - y2))                    # MANHATTAN DISTANCE
    h2 = max(abs(x1 - x2), (y1 - y2))                     # DIAGONAL DISTANCE
    h3 = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)           # EUCLIDIAN DISTANCE
    h4 =  math.sqrt(((x1 - x2)**2 + (y1 - y2)**2)/2)      # ADMISSIBLE HEURISTIC (RMS VALUE OF abs(x1 - x2) and abs(y1 - y2))
    h5 =  abs((x1 - x2)**2) + abs((y1 - y2)**2)           # NON - ADMISSIBLE HEURISTIC 
    h6 = 0                                                # DIJKSTRA'S 
    return h4

# function to find children nodes of a node
def children(image, current):
    children_list = []
    if (isValid(image, current)):
        if (key == False):                                               # case when diagonal movement is restricted

            children_list.append((current[0] + 1, current[1]))                
            children_list.append((current[0] - 1, current[1]))
            children_list.append((current[0], current[1] + 1))
            children_list.append((current[0], current[1] - 1))

        elif (key == True):                                              # case when diagonal movement is restricted
                                                                      
            children_list.append((current[0] + 1, current[1]))
            children_list.append((current[0] - 1, current[1]))
            children_list.append((current[0], current[1] + 1))
            children_list.append((current[0], current[1] - 1))
            children_list.append((current[0] + 1, current[1]+1))
            children_list.append((current[0] - 1, current[1] - 1))
            children_list.append((current[0] - 1, current[1] + 1))
            children_list.append((current[0] + 1, current[1] - 1))

    return children_list
        
def trace_path(image, cell_details, src, dstn):

    trackback = dstn
    path_cost = cell_details[dstn[0], dstn[1]].g                         # cost of path to reach from src to dstn = g value of dstn
    print(f"PATH_COST = {path_cost} units")
    # print(cell_details)
    while(trackback != src):
        image[trackback[0], trackback[1]] = PATH_COLOR
        trackback = cell_details[trackback[0], trackback[1]].parent
      
    image[src[0], src[1]] = SRC_COLOR  
    image[dstn[0], dstn[1]] = DSTN_COLOR 
    
    # image resized from (100, 100) to (1000, 1000)
    h, w = image.shape[:2]
    res = np.zeros((10*h, 10*w, 3), dtype = np.uint8)

    for i in range(10*h):
            for j in range(10*w):
                x=int(((i-(i%10))/10))
                y=int(((j-(j%10))/10))
                res[i][j][0]= image[x][y][0]
                res[i][j][1]= image[x][y][1]
                res[i][j][2]= image[x][y][2]

    cv.namedWindow("PATH", cv.WINDOW_AUTOSIZE)
    cv.imshow("PATH", res)
    k = cv.waitKey(0)
    if (k == ord('q')):
        cv.imwrite(r"Task 1_20CS30037\Diagonal_imgs\dijkstra.jpeg", res)
        cv.destroyAllWindows()
    
    return

def a_star(image, cell_details, src, dstn):

    time_1 = time.time()                                                   # time when A_star/dijkstra  begins searching

    foundDest = False                                                      # a bool to check if A-star/ dijkstra has found the path or not
    
    # initialsing open list
    open_list = []
    # initialising details of src node and appending it to open_list
    cell_details[src[0], src[1]].parent = src
    cell_details[src[0], src[1]].g = 0
    cell_details[src[0], src[1]].h = heuristic(src, dstn)
    cell_details[src[0], src[1]].f = cell_details[src[0], src[1]].g + cell_details[src[0], src[1]].h
    cell_details[src[0], src[1]].open = True
    
    heapq.heappush(open_list ,cell_details[src[0],src[1]])                 # using heap implementation, we can even interate the open_list unlike priority queue

    iterations = 0  
    while(len(open_list)):

        iterations += 1
        # heap implementation pops out the lowest f element
        current_node = heapq.heappop(open_list)
        pos = current_node.position
        cell_details[pos[0], pos[1]].open = False

        # # Visualisation
        # cv.namedWindow("Path_finding", cv.WINDOW_NORMAL)
        # cv.imshow('Path_finding', image)
        # k = cv.waitKey(1)
        # if (k == ord('q')):
        #     cv.destroyAllWindows
        
        # putting the current node in closed category if the current node not in closed category already
        if (not cell_details[pos[0], pos[1]].closed):
            current_node.closed = True
            cell_details[pos[0], pos[1]] = current_node

        # Nodes already visited by now (nodes in closed category) marked as VISITED_COLOR
        image[pos[0], pos[1]] = VISITED_COLOR

        # if dstn is found, return and show path
        if (current_node.position == dstn):
            foundDest = True
            print("The required path has been discovered as per the chosen heuristic.")
            time_2 = time.time()                                           # time when A-star/dijkstra  ends 
            print(f"Time taken by the program to run:\n {time_2 - time_1}")
            trace_path(image, cell_details, src, dstn)          
            return foundDest

        # accessing neighbours of the current node:
        children_list = children(image, pos)

        for child in children_list:
            x, y = child
            if (not isValid(image, (x, y))):
                continue                                                   # we neglect the node if it isnt a valid pixel
            
            # temporary cost estimations
            if (x == pos[0] or y == pos[1]):                               # for case of non_diagonal neighbours
                g_temp = cell_details[pos[0], pos[1]].g + 1
            else:                                                          # for case of diagonal neighbours
                g_temp = cell_details[pos[0], pos[1]].g + math.sqrt(2)
            

            if (g_temp < cell_details[x, y].g):                            # if new cost is lower, we update f, g, h and the parent 
                cell_details[x, y].closed = False
                cell_details[x, y].g = g_temp
                cell_details[x, y].h = heuristic((x, y), dstn)
                cell_details[x, y].f = g_temp + heuristic((x, y), dstn)
                cell_details[x, y].parent = cell_details[pos[0],pos[1]].position
                cell_details[x, y].open = True                             # adding the node to open list
                heapq.heappush(open_list, cell_details[x, y])
                image[x, y] = CURRENT_COLOR                                # node added in open - list currently are colored CURRENT_COLOR

            else:
                continue
    # print(iterations)
    return foundDest

def main():

    img_path= r"Task 1_20CS30037\Task_1_Low.png"
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    # cv.namedWindow('original image', cv.WINDOW_NORMAL)
    # cv.imshow('original image', img)
    # cv.waitKey(0)
    print(img.shape)
    h, w = img.shape[:2]
    src, dstn = src_dstn_find(img, h, w)
    
    # 2-d array to store info for all cells
    cell_details = np.empty((h, w), dtype = object)  
    for i in range(0, h):
        for j in range(0, w):
            cell_details[i][j] = Node(None, None)
            cell_details[i][j].position = (i, j)

    findDest = a_star(img, cell_details, src, dstn)
    
    if (findDest == False):
        print("The required optimum path could not be discovered.")
    return

if __name__ == "__main__":
    main()



