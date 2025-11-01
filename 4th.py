import numpy as np
import matplotlib.pyplot as plt

# Define the system matrix A as a np.array
# This is the matrix from Exercise 1
A = np.array([
    [3, -1, 0],
    [0, -4, 2],
    [0, 0, -1]
])

# Define the time vector for the simulation
t = np.linspace(0, 5, 500) # Increased points for a smoother curve

# --- Test with different initial conditions (x0) ---
# You can uncomment any of these to see the different behaviors.

# A general initial condition
x0 = np.array([1, 1, 1]) 

# An initial condition that lies on the stable subspace (a combination of v2 and v3)
# v2 = [1, 7, 0], v3 = [1, 4, 6]
# x0 = np.array([2, 11, 6]) # Example: v2 + v3

# An initial condition purely along the unstable eigenvector v1
# x0 = np.array([1, 0, 0])

# --- Compute the state evolution manually ---
# The solution to x_dot = Ax is x(t) = exp(At) * x0.
# This can be calculated using the eigenvalues and eigenvectors of A.
# x(t) = c1*v1*exp(λ1*t) + c2*v2*exp(λ2*t) + c3*v3*exp(λ3*t)
# where x0 = c1*v1 + c2*v2 + c3*v3

# Eigenvalues and Eigenvectors from Exercise 1
lmbda = np.array([3, -4, -1])
v1 = np.array([1, 0, 0])
v2 = np.array([1, 7, 0])
v3 = np.array([1, 4, 6])

# Matrix of eigenvectors V
V = np.array([v1, v2, v3]).T

# To find coefficients c = [c1, c2, c3], we solve x0 = Vc => c = V_inv * x0
V_inv = np.linalg.inv(V)
c = V_inv @ x0

# Compute the evolution of the state over time
# We create a list to hold the state vector x at each time step
x_t_list = []
for ti in t:
    # Calculate x(t) for the current time step ti
    term1 = c[0] * v1 * np.exp(lmbda[0] * ti)
    term2 = c[1] * v2 * np.exp(lmbda[1] * ti)
    term3 = c[2] * v3 * np.exp(lmbda[2] * ti)
    x_ti = term1 + term2 + term3
    x_t_list.append(x_ti)

# Convert the list of results into a NumPy array for easy plotting
x_t = np.array(x_t_list)

# --- Create separate plots for each state variable ---

plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle('Free Evolution of the Dynamical System', fontsize=16)

# Plot for x1(t)
axs[0].plot(t, x_t[:, 0], label='x1(t)', color='blue')
axs[0].set_title('State Variable x1')
axs[0].set_ylabel('Value')
axs[0].legend()

# Plot for x2(t)
axs[1].plot(t, x_t[:, 1], label='x2(t)', color='green')
axs[1].set_title('State Variable x2')
axs[1].set_ylabel('Value')
axs[1].legend()

# Plot for x3(t)
axs[2].plot(t, x_t[:, 2], label='x3(t)', color='red')
axs[2].set_title('State Variable x3')
axs[2].set_xlabel('Time (t)')
axs[2].set_ylabel('Value')
axs[2].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()