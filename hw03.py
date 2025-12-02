import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm




"""

Theory:

What are the state variables?

The state variables are the Lean Angle (φ) and the Lean Rate (φ˙). These correspond to the two elements in state vector x.
The first variable, x1 (x[0]), represents the physical orientation of the bike, specifically,
the angle at which it is tilted away from the vertical upright position. The second variable, x2 (x[1), 
represents the angular velocity, which tells us how fast the bike is currently falling toward the ground or swinging back upright.


What do you expect from the bicycle evolution?

Based on the physics of the "Neutral Model," we expect the simulation to demonstrate inherent instability.
Because the model treats the bike like an inverted pendulum with no self correcting steering geometry (since trail c=0),
gravity forces act as a positive feedback loop. This means that for any initial condition other than perfectly upright (0,0),
the lean angle (φ) will grow exponentially over time. We will see the bike start to tilt slowly,
then accelerate rapidly away from the vertical position until it hits the ground at ±90∘ (±π/2 radians),
at which point the code's bicycle_fall function will clamp the values to simulate a crash.


Why do you expect it staying upright or falling down?

We expect the bicycle to stay upright only in the theoretical case where it is perfectly vertical (0 degrees)
because the force of gravity pushes straight down through the wheels, creating zero rotational force (torque).
However, we expect it to fall in all other scenarios because the bike acts as an unstable inverted pendulum.
Once the Center of Mass shifts even slightly to the side, gravity pulls on it,
creating a torque that drags the bike further down.
Because this "Neutral Model" lacks the steering geometry (trail c=0) needed to naturally turn the wheel and
generate a restoring centrifugal force, there is nothing to fight gravity,
the lean creates more torque, which creates more lean, causing a runaway acceleration toward the ground.


"""




def main():
    m = 87 # mass [kg]
    a = 0.492  # center of mass coordinates along x [m]
    h = 1.028  # center of mass coordinates along z [m]
    b = 1 # wheel distance [m]
    c = 0 # trail [m]
    lambda_angle = np.pi / 2 # head angle [rad]
    J = m * h ** 2 # inertia xx [kg*m^2]
    D = m * a * h # inertia xz [kg*m^2]
    V = 4 # linear velocity of rear wheel [m/s] (positive for forward motion)
    g = 9.81 # gravity acceleration [m/s^2]

    ################################################################################
    ########################### FILL FROM HERE ON ##################################
    ################################################################################

    # dynamical matrix of the system 
    A = np.array([
        [0, 1],
        [g/h, 0]
    ])

    # time vector
    t = np.arange(0, 2.01, 0.01)





    # initial conditions: choose 3-5 initial conditions for the state variables
    # show how the system evolves, and what happens when the bicycle starts upright with no velocity

    # Perfectly Upright (0,0), The Equilibrium
    # Small Lean (0.1 rad), Unstable fall to the right
    # Small Lean (-0.1 rad), Unstable fall to the left

    init_conds = [
        np.array([0.0, 0.0]),
        np.array([0.1, 0.0]),
        np.array([-0.1, 0.0]),
        np.array([0.0, 0.5]),
        np.array([0.05, -0.2])
    ]

    for i, x0 in enumerate(init_conds):
        # Calculate free evolution
        x = free_evolution(A, t, x0)
        
        # Apply the ground constraint
        x = bicycle_fall(x, t)

        # Plot phi (Lean Angle)
        # We only plot column 0 (Angle), column 1 is Velocity
        plt.plot(t, x[:, 0], label=f'Init: [{x0[0]}, {x0[1]}]')


    plt.title('Free Evolution of Neutral Bicycle Model')
    plt.xlabel('Time [s]')
    plt.ylabel('Lean Angle [rad]')
    plt.axhline(y=np.pi/2, color='r', linestyle='--', alpha=0.3, label='Ground (Left/Right)')
    plt.axhline(y=-np.pi/2, color='r', linestyle='--', alpha=0.3)
    plt.grid(True)
    plt.legend()
    plt.show()

# free evolution function to be completed
def free_evolution(A, t, x0):
    # Prepare an empty array to hold state vectors for all time steps
    # Shape: (Number of time steps, 2 states)
    x_history = np.zeros((len(t), len(x0)))
    
    for i, time_step in enumerate(t):
        # Analytical Solution for Linear System: x(t) = e^(At) * x(0)
        # expm computes the matrix exponential
        x_history[i] = np.dot(expm(A * time_step), x0)
        
    return x_history

# bicycle falling: this function is given. Please explain in the following what does the function mean and what case(s) it is covering. The explanation(s) will be part of the evaluation
def bicycle_fall(x, t):
    samples = len(t)
    for i in range(samples):
        if x[i, 0] >= np.pi / 2:
            x[i:, 0] = np.pi / 2
            x[i:, 1] = 0
            break
        if x[i, 0] <= -np.pi / 2:
            x[i:, 0] = -np.pi / 2
            x[i:, 1] = 0
            break
    return x
# YOUR EXPLANATION OF `bicycle_fall` HERE: 

"""

The bicycle_fall function acts as a physical constraint simulator that stops the mathematical model from behaving unrealistically.
The equations of motion would allow the bicycle to spin endlessly like a propeller (leaning past 90 degrees to 180, 270, etc.) as the instability accelerates.
This function iterates through the simulation data to detect if the bike has hit the ground.
It covers two cases, falling to the right (where the lean angle is greater than or equal to 90 degrees) and
falling to the left (where the lean angle is less than or equal to -90 degrees). Once either limit is triggered,
the function overwrites all subsequent data points for the rest of the simulation time,
forcing the lean angle to stay constant at 90 degrees (lying flat) and the velocity to become zero, mimicking a crash


"""



if __name__ == "__main__":
    main()
