import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Define the system matrix A as a np.array
A = np.array([[3, -1, 0],
              [0, -4, 2],
              [0, 0, -1]])

# Define the input matrix B as a np.array
B = np.array([[1, 0],
              [0, 1],
              [0, 0]])



# Define the time vector
t = np.linspace(0, 5, 1000)

# Initial condition --> try with more than one!
x0 = np.array([0.5, 1, -0.5])

# Define the controller gain K
Ks = [1, 3, 5]

def system_dynamics(x, t, A_cl):
    # @ is the matrix multiplication operator in numpy
    return A_cl @ x


plt.figure(figsize=(10, 12))

plt.subplots_adjust(hspace=0.5)

# Create separate plots for each state variable
for i, k_val in enumerate(Ks):
    
    # Create closed loop system
    # Construct the K matrix (2x3) based on the scalar k
    # K = [[k, 0, 0],
    #      [0, k, 0]]
    K_matrix = np.array([[k_val, 0, 0],
                         [0, k_val, 0]])
    
    # Calculate Closed-Loop Matrix: A_cl = A - B*K
    A_cl = A - (B @ K_matrix)


    # odeint solves the differential equation defined by system_dynamics
    # odeint means "ordinary differential equation integration" 
    x_sol = odeint(system_dynamics, x0, t, args=(A_cl,))

    plt.subplot(3, 1, i+1)
    plt.plot(t, x_sol[:, 0], label='$x_1$ (Controlled)', linewidth=2)
    plt.plot(t, x_sol[:, 1], label='$x_2$ (Controlled)', linewidth=2)
    plt.plot(t, x_sol[:, 2], label='$x_3$ (Uncontrollable)', linestyle='--', alpha=0.7)
    # x_sol[:, 0] gives the first column of x_sol
    # x_sol[:, 1] gives the second column of x_sol
    # x_sol[:, 2] gives the third column of x_sol


    plt.title(f'System Response with k = {k_val}')
    plt.xlabel('Time (s)')
    plt.ylabel('State Amplitude')
    plt.grid(True)
    plt.legend()


plt.show()
