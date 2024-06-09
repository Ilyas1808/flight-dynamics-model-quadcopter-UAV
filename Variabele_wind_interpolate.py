import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from drone import Quadcopter
from pid import PID_Controller

#------------------------------------------------------------------------------#

#Data gehaald van Meteomatics API
#link: api.meteomatics.com/2024-03-05T00:00:00Z--2024-03-08T00:00:00Z:PT1H/wind_gust_10m_24h:ms,wind_dir_10m:d,wind_speed_10m:ms/52.520551,13.461804/html
#User: student_elmadani_ilyas
#Password: BXP82qf09f
#localisatie moet veranderd worden in functie van gevraagde stad
#wind_dir is in degrees 
data = """
validdate;wind_speed_10m:ms;wind_gusts_10m_24h:ms;wind_dir_10m:d
1:00;1.7;10.2;327.5
2:00;1.6;9;285.0
3:00;1.5;11;279.5
4:00;1.5;10.4;285.6
5:00;1.4;8;275.9
6:00;1.4;6;282.1
7:00;1.6;10;270.1
"""

# Parse data, alles splitsen voor in arrays te kunnen plaatsen
lines = data.strip().split('\n')
headers = lines[0].split()
data = [line.split(";") for line in lines[1:]]
hours = [entry[0] for entry in data]
bearing = [float(entry[3]) for entry in data]
wind_speed = [float(entry[1]) for entry in data]
wind_gusts = [float(entry[2]) for entry in data]

# Uren moeten naar integers worden omgezet 
hours_int = [int(hour.split(':')[0]) for hour in hours]

# Berkening van windsnleheid en gust, belangrijk voor het plotten 
time = np.linspace(0, 1, len(hours))  # interval voor elk uur  
wind_speed_curve = np.array(wind_speed)
wind_gusts_curve = np.array(wind_gusts)


#simulatie_parameters
sim_start = 0
number_of_hours = hours_int[-1]  #aantal uren is gelijk aan het laatste getal van uur 
print("NUMBER OF HOURS",number_of_hours)
sim_end = number_of_hours*60 
print("SIMULATION time",sim_end)



interpolated_wind = []
true_bearing = []
counter=0
for i in range(len(hours)-1):
    counter+=1
    start_wind = wind_speed_curve[i]
    end_wind = wind_speed_curve[i+1]
    max_gust = wind_gusts_curve[i]
    t = np.linspace(0, 2, (sim_end*2)+1)  
    angle = np.linspace(bearing[0], bearing[1], (sim_end*2)+1)
    cosine_curve = (1 - np.cos(np.pi * t)) / 2  # Cosinus curve
    interpolated_wind.extend(start_wind + cosine_curve * (max_gust - start_wind))
    true_bearing.extend(angle)

print("LENGTH OF TIME",len(t))
# Plot
plt.figure(figsize=(10.6,4))
plt.plot(np.linspace(0, hours_int[-1], len(interpolated_wind)), interpolated_wind, label='Windsnelheid (m/s)')
plt.xlabel('Tijd (Uur)')
plt.ylabel('Windsnelheid (m/s)')
plt.title('Geïnterpoleerde windsnelheid over 6 uur')
plt.legend()
plt.legend(loc='upper left')
#plt.xticks(range(0, 25))
plt.grid(True)
#plt.tight_layout()
plt.show()


######################################
dt = 0.1
Kp_pos = [.95, .95, 12]
Kd_pos = [2, 1.5, 19]
Ki_pos = [0.2, 0.2, 1.0]
Ki_sat_pos = 1.1 * np.ones(3)
Kp_ang = [7, 7, 25]
Kd_ang = [4, 4, 9.]
Ki_ang = [0.1, 0.1, 0.1]
time_index = np.arange(sim_start, sim_end + dt, dt)
wind = []
deviation = 10
ang_vel = np.array([0.0, 0.0, 0.0])
ref = np.array([50., 60., 70])
pos = [0.5, -0.5, 0.]
vel = np.array([0., 0., 0.])
ang = np.array([0., 0., 0.])
ang_vel_init = ang_vel.copy()
gravity = 9.8
Ki_sat_ang = 0.1 * np.ones(3)

quadcopter = Quadcopter(pos, vel, ang, ang_vel, ref, dt)
pos_controller = PID_Controller(Kp_pos, Kd_pos, Ki_pos, Ki_sat_pos, dt)
angle_controller = PID_Controller(Kp_ang, Kd_ang, Ki_ang, Ki_sat_ang, dt)

total_error = []
position_total = []
total_thrust = []

def initialize_results(res_array, num):
    for i in range(num):
        res_array.append([])

position = []
initialize_results(position,3)
velocity = []
initialize_results(velocity,3)
angle = []
initialize_results(angle,3)
angle_vel = []
initialize_results(angle_vel,3)
motor_thrust = []
initialize_results(motor_thrust,4)
body_torque = []
initialize_results(body_torque,3)
i = 0
print("Interpolated Bearing",len(interpolated_wind),len(true_bearing),len(time_index))
for time in enumerate(time_index):
    wind_x = interpolated_wind[i]*math.cos(true_bearing[i])
    wind_y = interpolated_wind[i]*math.sin(true_bearing[i])
    wind = [wind_x,wind_y,0]
#    print(wind)
    pos_error = quadcopter.calc_pos_error(quadcopter.pos)
    vel_error = quadcopter.calc_vel_error(quadcopter.vel)
    des_acc = pos_controller.control_update(pos_error, vel_error)
    des_acc[2] = (gravity + des_acc[2]) / (math.cos(quadcopter.angle[0]) * math.cos(quadcopter.angle[1]))

    thrust_needed = quadcopter.mass * des_acc[2]
    mag_acc = np.linalg.norm(des_acc)
    if round(mag_acc) == 0:
        mag_acc = 1
    print("TESTING",mag_acc, des_acc[1] ,quadcopter.angle[1],des_acc[0])
    ang_des = [math.asin(-des_acc[1] / mag_acc / math.cos(quadcopter.angle[1])),
               math.asin(des_acc[0] / mag_acc), 0]
    mag_angle_des = np.linalg.norm(ang_des)
    if mag_angle_des > quadcopter.max_angle:
        ang_des = (ang_des / mag_angle_des) * quadcopter.max_angle

    quadcopter.angle_ref = ang_des
    ang_error = quadcopter.calc_ang_error(quadcopter.angle)
    ang_vel_error = quadcopter.calc_ang_vel_error(quadcopter.ang_vel)
    tau_needed = angle_controller.control_update(ang_error, ang_vel_error)

    quadcopter.des2speeds(thrust_needed, tau_needed)
    quadcopter.step(wind)

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

    i+=1
def total_plot():
    fig = plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')

    axes = fig.add_subplot(3, 2, 1)
    axes.plot(time_index, position[0], label='x')
    axes.plot(time_index, position[1], label='y')
    axes.set_title('Postions')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('position (m)')
    axes.legend()

    axes = fig.add_subplot(3, 2, 2)
    axes.plot(time_index, position[2], label='z')
    axes.set_title('Vertical Position')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('altitude (m)')

    axes = fig.add_subplot(3, 2, 3)
    axes.plot(time_index, velocity[0], label='x_dot')
    axes.plot(time_index, velocity[1], label='y_dot')
    axes.plot(time_index, velocity[2], label='z_dot')
    axes.set_title('Linear Velocity')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('velocity (m/min)')
    axes.legend()

    axes = fig.add_subplot(3, 2, 4)
    axes.plot(time_index, angle[0], label='roll')
    axes.plot(time_index, angle[1], label='pitch')
    axes.plot(time_index, angle[2], label='yaw')
    axes.set_title('Angles')
    axes.set_xlabel('time (mins)')
    axes.set_ylabel('angle (deg)')
    axes.legend()

    plt.tight_layout(pad=0.4, w_pad=1, h_pad=1)
    plt.show()
    
    fig2 = plt.figure(figsize=(16, 8), dpi=100).add_subplot(projection='3d')
    fig2.plot(position[0], position[1], position[2])
    fig2.set_title('Flight Path')
    fig2.set_xlabel('x (m)')
    fig2.set_ylabel('y (m)')
    fig2.set_zlabel('z (m)')
    plt.show()


total_plot()
