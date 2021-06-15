import numpy as np
from numpy import dot
import matplotlib.pyplot as plt

file1= open(r"Task 3_20CS30037\Kalman_data.txt", "r")
data_string = file1.readlines()

global data_table
data_table = np.full((360,4), 0, dtype = float)

data_string[0] = data_string[0].split(",")
for j in range(0,2):
    data_table[0][j] = float(data_string[0][j])
    
for j in range (2, 4):                             # initial velocities in x and y dirn = 0
    data_table[0][j] = float(0)
    

for i in range (1, len(data_string)):
    data_string[i] = data_string[i].split(" , ")
    for j in range (0, 4):
        data_table[i][j] = float(data_string[i][j])

file1.close()

# Definition and initialsiation of various matrices of  the KALMAN FILTER:
# 1.) State Matrix => X_k
X_k = np.full((4,1), 0, dtype = float)

# 2.) Predicted State Matrix => X_k_p
X_k_p = np.full((4,1), 0, dtype = float)

# 3.) State Measurement Matrix => Y_k
Y_k = np.full((4,1), 0, dtype = float)

# 4.) Process Covariance Matrix (represents errors in the estimate) => P_k
P_k = np.full((4,4), 0, dtype = float)

# 5.) Control Variable Matrix (to incorporate the change in state of the object due to factors such as acceleration) => U_k
ax  = 0
ay = 0
U_k = np.array([[ax], [ay]])

# 6.) Predicted State Noise Matrix (to incorporate any computational errors by the device in the process of prediction/estimation of a state) => W_k
W_k = np.full((4,1), 0, dtype = float)
W_k[0, 0] = 0.001
W_k[1, 0] = 0.001
W_k[1, 0] = 0.01
W_k[1, 0] = 0.01

# 7.) Process Noise Covarience Matrix => Q_k (prevents elements  of state covariance matrix from becoming too small or achieving value = 0)
Q_k = np.zeros((4,4), dtype = float)
Q_k[0, 0] = 0.01
Q_k[1, 1] = 0.01
Q_k[2, 2] = 0.001
Q_k[3, 3] = 0.001

# 8.) Kalman Gain Matrix => K
K = np.full((4,4), 0, dtype = float)

# 9.) Sensor Noise Covarience Matrix (to take into account the uncertainitites and errors in the measurement process)=> R
R = np.zeros((4, 4), dtype = float)
R[0, 0] = 0.01
R[1, 1] = 0.01
R[2, 2] = 0.001
R[3, 3] = 0.001

# 10.) Adjustment matrices A, B, H:
# Considering time constant = 1 second
delt = float(1)
A = np.array([[1, 0, delt, 0], [0, 1,0, delt], [0, 0, 1, 0], [0, 0, 0, 1]])
B = np.array([[0.5*((delt)**2), 0], [0, 0.5*((delt)**2)], [delt, 0], [0, delt]])

# H matrix = Identity Matrix in our problem
H = np.identity(4, dtype = float)
# 11.)  Identity Matrix = I
I = np.identity(4, dtype = float)

# Initial State Covarience Matrix:
P_k[0, 0] = 0.2       # sdx**2
P_k[1, 1] = 0.15      # sdy**2
P_k[2, 2] = 0.1       # sdvx**2
P_k[3, 3] = 0.07      # sdvy**2

# Step1 => Updation of the State Matrix and Process Covarience Matrix:
def stateUpdate(A, B, U_k, X_k_p, W_k, P_k, Q_k):
    # X_k_p = A*(X_k-1_p)  + B*U_k + W_k
    X_k_p= dot(A, X_k_p) + dot(B, U_k) + W_k
    # P_k_p = A*(P_k-1)*A.T + Q_K
    P_k = dot(A,dot(P_k, A.T)) + Q_k
    return (X_k_p, P_k)

# Step2 => Calculation of the Kalman Gain:
def kalmanGain(P_k, H, R):
    # K = ((P_k_p)*(H.T))/((H*(P_k_p)*(H.T)) + R)
    K = dot(P_k, dot(H.T, np.linalg.inv(R + dot(H, dot(P_k, H.T)))))
    return K

# Step3 => Calculating the  updated state matrix using the Kalman gain:
def currentState(X_k, X_k_p, K, Y_k, H):
    # X_k= X_k_p + K*(Y_k - H*(X_k_p))
    X_k = X_k_p + dot(K, (Y_k - dot(H, X_k_p)))
    return X_k

# Step4 => Updating the process covarience matrix:
def CovUpdate(P_k, H, I, K):
    # P_k = (I - K*H)*(P_k_p)
    P_k = dot((I - dot(K, H)), P_k)
    for i in range (0, 4):                        
        for j in range (0, 4):
            if (i != j):
                P_k[i][j] = 0
    return P_k

res_state = np.full((360,4), 0, dtype = float)                     # 2-D array to store the resultant state matrix after each iteration
res_cov = np.full((360, 4, 4), 0, dtype = float)                   # 3-D array to store the resultant covariance matrix after each iteration

# Initial predicted state matrix:
X_k_p = [[data_table[0][0]],[data_table[0][1]], [0], [0]]

file2 = open(r"Task 3_20CS30037\Output.txt", "w")

# list of estimated coordinates to be used later for plotting the trajectory, to compare b/w measured, estimated and Kalman coordinates
x_coord_estim = []
y_coord_estim = []


for i in range (0, data_table.shape[0]):
    X_k_p, P_k = stateUpdate(A, B, U_k, X_k_p, W_k, P_k, Q_k)
    
    # appending the estimated x and y coordinates after each iteration to a list in ordeer to plot at the end for comparison
    x_coord_estim.append(X_k_p[0][0])
    y_coord_estim.append(X_k_p[1][0])
    
    K = kalmanGain(P_k, H, R)
    Y_k = np.reshape(data_table[i], (4,1))
    X_k = currentState(X_k, X_k_p, K, Y_k, H)
    res_state[i] = np.reshape(X_k, (1,4))
    P_k = CovUpdate(P_k, H, I, K)
    res_cov[i] = P_k
    X_k_p = X_k
    
    print(f"For state {i} :\nUpdated State Matrix:\n{res_state[i]}\nUpdated Covarience Matrix:\n{res_cov[i]}\n\n")
    L = [f"For state {i} :\nUpdated State Matrix:\n{res_state[i]}\nUpdated Covarience Matrix:\n{res_cov[i]}\n\n"]
    file2.writelines(L)

file2.close()


# Graph Plotting:
x_coord_res = []
y_coord_res = []

x_coord_meas = []
y_coord_meas = []

for i in range (0, 360):
        
        x_coord_res.append(res_state[i][0])
        y_coord_res.append(res_state[i][1])

        x_coord_meas.append(data_table[i][0])
        y_coord_meas.append(data_table[i][1])


figure, graph = plt.subplots(2,2)

graph[0, 0].plot(x_coord_res, y_coord_res, color = 'green')
graph[0, 0].plot(x_coord_meas, y_coord_meas, color = 'red')
graph[0, 0].plot(x_coord_estim, y_coord_estim, color = 'blue')
graph[0, 0].title.set_text("Combined Graph")

graph[0, 1].plot(x_coord_res, y_coord_res, color = 'green')
graph[0, 1].title.set_text("Final Result after Kalman Filter")

graph[1, 0].plot(x_coord_meas, y_coord_meas, color = 'red')
graph[1, 0].title.set_text("Measured Data")

graph[1, 1].plot(x_coord_estim, y_coord_estim, color = 'blue')
graph[1, 1].title.set_text("Estimated Data")

plt.show()
