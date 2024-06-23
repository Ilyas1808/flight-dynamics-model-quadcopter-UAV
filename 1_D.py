import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from drone import Quadcopter
from pid import PID_Controller

# Simulation parameters
sim_start = 0
sim_end = 30
dt = 0.01


# Position controller gains
Kp_pos = [.95, .95, 15.] # proportional [x,y,z]
Kd_pos = [1.8, 1.8, 15.]  # derivative [x,y,z]
Ki_pos = [0.2, 0.2, 1.0] # integral [x,y,z]
Ki_sat_pos = 1.1*np.ones(3)  # saturation for integral controller (prevent windup) [x,y,z]

# Gains for angle controller
Kp_ang = [6.9, 6.9, 25.] # proportional [x,y,z]
Kd_ang = [3.7, 3.7, 9.]  # derivative [x,y,z]
Ki_ang = [0.1, 0.1, 0.1] # integral [x,y,z]
Ki_sat_ang = 0.1*np.ones(3)  # saturation for integral controller (prevent windup) [x,y,z]


# Time array
time_index = np.arange(sim_start, sim_end + dt, dt)

# Wind parameters
wind = [10, 5., 0.]

wind_x = np.full_like(time_index, wind[0])
wind_y = np.full_like(time_index, wind[1])

# Initial conditions
#deviation = 10 # intial perturbation
ang_vel = np.array([0.0, 0.0, 0.0]) # initial angular velocity
ref = np.array([10., 20., 10.]) # target position 
pos = [10., 20., 10.] # initial position
vel = np.array([0., 0., 0.]) # initial velocity
ang = np.array([0., 0., 0.]) # initial Angle
ang_vel_init = ang_vel.copy()
gravity = 9.81

# Initialize Quadcopter and Controllers
quadcopter = Quadcopter(pos, vel, ang, ang_vel, ref, dt)
pos_controller = PID_Controller(Kp_pos, Kd_pos, Ki_pos, Ki_sat_pos, dt)
angle_controller = PID_Controller(Kp_ang, Kd_ang, Ki_ang, Ki_sat_ang, dt)

# Initialize result arrays
total_error = []
position_total = []
total_thrust = []

plt.figure(figsize=(10.6, 4))
plt.plot(time_index, wind_x, label='Windsnelheid in x (m/s)')
plt.plot(time_index, wind_y, label='Windsnelheid in y (m/s)')
plt.xlabel('Tijd (s)')
plt.ylabel('Windsnelheid (m/s)')
plt.title('Windsnelheid over tijd')
plt.legend()
plt.grid(True)
plt.show()

def initialize_results(res_array, num):
    """Initialize empty lists in result array."""
    for i in range(num):
        res_array.append([])

# Initialize result arrays for position, velocity, angle, etc.
position = []
initialize_results(position, 3)
velocity = []
initialize_results(velocity, 3)
angle = []
initialize_results(angle, 3)
angle_vel = []
initialize_results(angle_vel, 3)
motor_thrust = []
initialize_results(motor_thrust, 4)
body_torque = []
initialize_results(body_torque, 3)

# Simulation loop
for time in enumerate(time_index):
    # Calculate position and velocity errors
    pos_error = quadcopter.calc_pos_error(quadcopter.pos)
    vel_error = quadcopter.calc_vel_error(quadcopter.vel)
    
    # Update position controller
    des_acc = pos_controller.control_update(pos_error, vel_error)
    des_acc[2] = (gravity + des_acc[2]) / (math.cos(quadcopter.angle[0]) * math.cos(quadcopter.angle[1]))
    
    # Calculate thrust needed
    thrust_needed = quadcopter.mass * des_acc[2]
    mag_acc = np.linalg.norm(des_acc)
    
    # Ensure non-zero magnitude to avoid division by zero
    if mag_acc == 0:
        mag_acc = 1
    
    # Calculate desired angles
    ang_des = [math.asin(-des_acc[1] / mag_acc / math.cos(quadcopter.angle[1])),
               math.asin(des_acc[0] / mag_acc), 0]
    mag_angle_des = np.linalg.norm(ang_des)
    
    # Limit desired angles to maximum angle
    if mag_angle_des > quadcopter.max_angle:
        ang_des = (ang_des / mag_angle_des) * quadcopter.max_angle
    
    # Update angle controller
    quadcopter.angle_ref = ang_des
    ang_error = quadcopter.calc_ang_error(quadcopter.angle)
    ang_vel_error = quadcopter.calc_ang_vel_error(quadcopter.ang_vel)
    tau_needed = angle_controller.control_update(ang_error, ang_vel_error)
    
    # Calculate motor speeds
    quadcopter.des2speeds(thrust_needed, tau_needed)
    
    # Step in time
    quadcopter.step(wind)
    
    # Record key attributes
    position_total.append(np.linalg.norm(quadcopter.pos))
    
    position[0].append(quadcopter.pos[0])
    position[1].append(quadcopter.pos[1])
    position[2].append(quadcopter.pos[2])
    
    velocity[0].append(quadcopter.vel[0])
    velocity[1].append(quadcopter.vel[1])
    velocity[2].append(quadcopter.vel[2])
    
    angle[0].append(np.rad2deg(quadcopter.angle[0]))
    angle[1].append(np.rad2deg(quadcopter.angle[1]))
    angle[2].append(np.rad2deg(quadcopter.angle[2]))
    
    angle_vel[0].append(np.rad2deg(quadcopter.ang_vel[0]))
    angle_vel[1].append(np.rad2deg(quadcopter.ang_vel[1]))
    angle_vel[2].append(np.rad2deg(quadcopter.ang_vel[2]))
    
    motor_thrust[0].append(quadcopter.speeds[0] * quadcopter.kt)
    motor_thrust[1].append(quadcopter.speeds[1] * quadcopter.kt)
    motor_thrust[2].append(quadcopter.speeds[2] * quadcopter.kt)
    motor_thrust[3].append(quadcopter.speeds[3] * quadcopter.kt)
    
    body_torque[0].append(quadcopter.tau[0])
    body_torque[1].append(quadcopter.tau[1])
    body_torque[2].append(quadcopter.tau[2])
    
    total_thrust.append(quadcopter.kt * np.sum(quadcopter.speeds))

def total_plot():
    """Plot various aspects of the simulation."""
    # Create subplots
    fig = plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
    axes = fig.add_subplot(3, 2, 1)
    axes.plot(time_index, position[0], label='x')
    axes.plot(time_index, position[1], label='y')
    axes.set_title('Posities')
    axes.set_xlabel('tijd (s)')
    axes.set_ylabel('positie (m)')
    axes.legend()

    axes = fig.add_subplot(3, 2, 2)
    axes.plot(time_index, position[2], label='z')
    axes.set_title('Verticale Positie')
    axes.set_xlabel('tijd (s)')
    axes.set_ylabel('hooghte (m)')

    axes = fig.add_subplot(3, 2, 3)
    axes.plot(time_index, velocity[0], label='Vx')
    axes.plot(time_index, velocity[1], label='Vy')
    axes.plot(time_index, velocity[2], label='Vz')
    axes.set_title('Lineaire snelheid')
    axes.set_xlabel('tijd (s)')
    axes.set_ylabel('snelheid (m/s)')
    axes.legend()

    axes = fig.add_subplot(3, 2, 4)
    axes.plot(time_index, angle[0], label='roll')
    axes.plot(time_index, angle[1], label='pitch')
    axes.plot(time_index, angle[2], label='yaw')
    axes.set_title('Hoeken')
    axes.set_xlabel('tijd (s)')
    axes.set_ylabel('hoek (deg)')
    axes.legend()

    plt.tight_layout(pad=0.4, w_pad=1, h_pad=1)
    plt.show()

    # 3D Flight Path
    fig2 = plt.figure(figsize=(16, 8), dpi=100).add_subplot(projection='3d')
    fig2.plot(position[0], position[1], position[2])
    fig2.set_title('Vliegbaan')
    fig2.set_xlabel('x (m)')
    fig2.set_ylabel('y (m)')
    fig2.set_zlabel('z (m)')
    plt.show()


total_plot()
