import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from drone import Quadcopter
from pid import PID_Controller

# csv van "csv_genereren.py" importen om deze data te gebruiken
flow_field = np.loadtxt('simulated_flow_field_hn.csv', delimiter=',')

# Pad selecteren in csv
wind_data_for_simulation = flow_field[0, :]

#wind = 0 voor baan zonder wind
wind_zero = np.zeros(len(wind_data_for_simulation))  

# Wind data plotten
plt.figure(figsize=(10.6, 4))
plt.plot(np.linspace(0, len(wind_data_for_simulation) * 0.1, len(wind_data_for_simulation)), wind_data_for_simulation, label='Wind Speed (m/s)')
plt.xlabel('Tijd (s)')
plt.ylabel('Windsnelheid (m/s)')
plt.title('Geïnterpoleerde windsnelheid over tijd')
plt.legend()
plt.grid(True)
plt.show()

# PID waarden, gewenste posities voor wind data MET wind 
dt = 0.1
Kp_pos = [.95, .95, 12] #[.95, .95, 12] zijn de beste waarden
Kd_pos = [2, 1.5, 19] #[2, 1.5, 19]
Ki_pos = [0.2, 0.2, 1.0] #[0.2, 0.2, 1.0]
Ki_sat_pos = 1.1 * np.ones(3)
Kp_ang = [7, 7, 25] #[7, 7, 25]
Kd_ang = [4, 4, 9.] #[4, 4, 9.]
Ki_ang = [0.1, 0.1, 0.1] # [0.1, 0.1, 0.1]
ang_vel = np.array([0.0, 0.0, 0.0])
target_position = np.array([[15., 10., 75], [45., 25., 75], [75., 50., 75], [100., 25., 75]])   
pos = [0., 10., 0.]
vel = np.array([0., 0., 0.])
ang = np.array([0., 0., 0.])
ang_vel_init = ang_vel.copy()
gravity = 9.81
Ki_sat_ang = 0.1 * np.ones(3)

total_error = []
position_total = []
total_thrust = []


#Lege lijsten maken om deze te kunnen vullen
def initialize_results(res_array, num):
    for i in range(num):
        res_array.append([])

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


# Tijdstappen
time_index = np.arange(0, len(wind_data_for_simulation) * dt, dt)
location_change_time = np.array_split(time_index, len(target_position))

i = 0
count = 0

for ref in target_position:
    quadcopter = Quadcopter(pos, vel, ang, ang_vel, ref, dt) #huidige posities, snelheden,... bijhouden in de quadcopter
    pos_controller = PID_Controller(Kp_pos, Kd_pos, Ki_pos, Ki_sat_pos, dt) #PID voor positie
    angle_controller = PID_Controller(Kp_ang, Kd_ang, Ki_ang, Ki_sat_ang, dt) #PID voor hoeken

    for t in location_change_time[count]:
        wind_speed = wind_data_for_simulation[i] #wind uit data
        wind = [wind_speed, wind_speed, 0] #wind in x  en y richting
        pos_error = quadcopter.calc_pos_error(quadcopter.pos) #verschil tussen huidige positie en gewenste positie
        vel_error = quadcopter.calc_vel_error(quadcopter.vel) #same maar voor snelheid
        des_acc = pos_controller.control_update(pos_error, vel_error) #wordt aangepast door positie en snelheidsfouten
        des_acc[2] = (gravity + des_acc[2]) / (math.cos(quadcopter.angle[0]) * math.cos(quadcopter.angle[1])) #gewenste versnelling

        thrust_needed = quadcopter.mass * des_acc[2] #thrust
        mag_acc = np.linalg.norm(des_acc) #magnitude van vernselling
        if round(mag_acc) == 0:
            mag_acc = 1
        ang_des = [math.asin(-des_acc[1] / mag_acc / math.cos(quadcopter.angle[1])),
                   math.asin(des_acc[0] / mag_acc), 0] #gewenste versnelling hangt ook af van de hoeken
        mag_angle_des = np.linalg.norm(ang_des)
        if mag_angle_des > quadcopter.max_angle:
            ang_des = (ang_des / mag_angle_des) * quadcopter.max_angle #gewenste versnelling dan met maximale hoek

        quadcopter.angle_ref = ang_des
        ang_error = quadcopter.calc_ang_error(quadcopter.angle) #fout in hoeken, verschil tussen huidige en gewenste hoek
        ang_vel_error = quadcopter.calc_ang_vel_error(quadcopter.ang_vel) # verschil tussen huidige en gewenste hoeksnelheid
        tau_needed = angle_controller.control_update(ang_error, ang_vel_error)

        quadcopter.des2speeds(thrust_needed, tau_needed) #berekenen van nodige motorsnelheid voor nodige thrust en koppel
        quadcopter.step(wind) #positie, sneleheid,... bijwerken in functie van thrust, koppel en wind

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

        i += 1
    count += 1
    pos = ref
    vel = np.array([quadcopter.vel[0], quadcopter.vel[1], quadcopter.vel[2]])
    ang = np.array([quadcopter.angle[0], quadcopter.angle[1], quadcopter.angle[2]])

# time_index aanpassen aan lengte van data
time_index = time_index[:len(position[0])]



###################################    GEEN WIND, zelfde code      ###################################
dt = 0.1
Kp_pos = [.95, .95, 12]
Kd_pos = [2, 2, 19] #2, 1.5, 19
Ki_pos = [0.2, 0.2, 1.0] #0.2, 0.2, 1.0
Ki_sat_pos = 1.1 * np.ones(3)
Kp_ang = [7, 7, 25]
Kd_ang = [4, 4, 9.]
Ki_ang = [0.1, 0.1, 0.1]
deviation = 10
ang_vel = np.array([0.0, 0.0, 0.0])
target_position = np.array([[15., 10., 75], [45., 25., 75], [75., 50., 75], [100., 25., 75]])   # Target Position  
pos = [0., 10., 0.]
vel = np.array([0., 0., 0.])
ang = np.array([0., 0., 0.])
ang_vel_init = ang_vel.copy()
gravity = 9.81
Ki_sat_ang = 0.1 * np.ones(3)

total_error = []
position_total = []
total_thrust = []

def initialize_results(res_array, num):
    for i in range(num):
        res_array.append([])

position1 = []
initialize_results(position1, 3)
velocity1 = []
initialize_results(velocity1, 3)
angle1 = []
initialize_results(angle1, 3)
angle_vel1 = []
initialize_results(angle_vel1, 3)
motor_thrust1 = []
initialize_results(motor_thrust1, 4)
body_torque1 = []
initialize_results(body_torque1, 3)

# Tijdstappen
time_index = np.arange(0, len(wind_zero) * dt, dt)
location_change_time = np.array_split(time_index, len(target_position))

i = 0
count = 0

for ref in target_position:
    quadcopter = Quadcopter(pos, vel, ang, ang_vel, ref, dt)
    pos_controller = PID_Controller(Kp_pos, Kd_pos, Ki_pos, Ki_sat_pos, dt)
    angle_controller = PID_Controller(Kp_ang, Kd_ang, Ki_ang, Ki_sat_ang, dt)

    for t in location_change_time[count]:
        wind_speed_zero = wind_zero[i] 
        wind_nul = [wind_speed_zero, wind_speed_zero, 0]
        pos_error = quadcopter.calc_pos_error(quadcopter.pos)
        vel_error = quadcopter.calc_vel_error(quadcopter.vel)
        des_acc = pos_controller.control_update(pos_error, vel_error)
        des_acc[2] = (gravity + des_acc[2]) / (math.cos(quadcopter.angle[0]) * math.cos(quadcopter.angle[1]))

        thrust_needed = quadcopter.mass * des_acc[2]
        mag_acc = np.linalg.norm(des_acc)
        if round(mag_acc) == 0:
            mag_acc = 1
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
        quadcopter.step(wind_nul) 

        position_total.append(np.linalg.norm(quadcopter.pos))

        position1[0].append(quadcopter.pos[0]) 
        position1[1].append(quadcopter.pos[1])
        position1[2].append(quadcopter.pos[2])

        velocity1[0].append(quadcopter.vel[0])
        velocity1[1].append(quadcopter.vel[1])
        velocity1[2].append(quadcopter.vel[2])

        angle1[0].append(np.rad2deg(quadcopter.angle[0]))
        angle1[1].append(np.rad2deg(quadcopter.angle[1]))
        angle1[2].append(np.rad2deg(quadcopter.angle[2]))

        angle_vel1[0].append(np.rad2deg(quadcopter.ang_vel[0]))
        angle_vel1[1].append(np.rad2deg(quadcopter.ang_vel[1]))
        angle_vel1[2].append(np.rad2deg(quadcopter.ang_vel[2]))

        motor_thrust1[0].append(quadcopter.speeds[0] * quadcopter.kt)
        motor_thrust1[1].append(quadcopter.speeds[1] * quadcopter.kt)
        motor_thrust1[2].append(quadcopter.speeds[2] * quadcopter.kt)
        motor_thrust1[3].append(quadcopter.speeds[3] * quadcopter.kt)

        body_torque1[0].append(quadcopter.tau[0])
        body_torque1[1].append(quadcopter.tau[1])
        body_torque1[2].append(quadcopter.tau[2])

        total_thrust.append(quadcopter.kt * np.sum(quadcopter.speeds))

        i += 1
    count += 1
    pos = ref
    vel = np.array([quadcopter.vel[0], quadcopter.vel[1], quadcopter.vel[2]])
    ang = np.array([quadcopter.angle[0], quadcopter.angle[1], quadcopter.angle[2]])


time_index = time_index[:len(position1[0])]





########################################### ALLES PLOTTEN  #####################################
def total_plot():
    fig = plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')

    axes = fig.add_subplot(3, 2, 1)
    axes.plot(time_index, position[0], label='x (wind)', color='red')
    axes.plot(time_index, position1[0], label='x (geen wind)', color='blue')
    axes.plot(time_index, position[1], label='y (wind)', color='black')
    axes.plot(time_index, position1[1], label='y (geen wind)', color='green')
    axes.set_title('Posities')
    axes.set_xlabel('tijd (s)')
    axes.set_ylabel('positie (m)')
    axes.legend()

    axes = fig.add_subplot(3, 2, 2)
    axes.plot(time_index, position[2], label='z (wind)', color='red')
    axes.plot(time_index, position1[2], label='z (geen wind)', color='blue')
    axes.set_title('Verticale positie')
    axes.set_xlabel('tijd (s)')
    axes.set_ylabel('hoogte (m)')

    axes = fig.add_subplot(3, 2, 3)
    axes.plot(time_index, velocity[0], label='x_dot (wind)', color='red')
    axes.plot(time_index, velocity1[0], label='x_dot (geen wind)', color='blue')
    axes.plot(time_index, velocity[1], label='y_dot (wind)', color='black')
    axes.plot(time_index, velocity1[1], label='y_dot (geen wind)', color='green')
    axes.plot(time_index, velocity[2], label='z_dot (wind)', color='pink')
    axes.plot(time_index, velocity1[2], label='z_dot (geen wind)', color='brown')
    axes.set_title('Lineare snelheid')
    axes.set_xlabel('tijd (s)')
    axes.set_ylabel('snelheid (m/s)')
    axes.legend()

    axes = fig.add_subplot(3, 2, 4)
    axes.plot(time_index, angle[0], label='roll (wind)', color='red')
    axes.plot(time_index, angle1[0], label='roll (geen wind)', color='blue')
    axes.set_title('Hoeken')
    axes.set_xlabel('tijd (sec)')
    axes.set_ylabel('hoek (deg)')
    axes.legend()

    axes = fig.add_subplot(3, 2, 5)
    axes.plot(time_index, angle[1], label='pitch (wind)', color='red')
    axes.plot(time_index, angle1[1], label='pitch (geen wind)', color='blue')
    axes.set_title('Hoeken')
    axes.set_xlabel('tijd (s)')
    axes.set_ylabel('hoek (deg)')
    axes.legend()
    
    axes = fig.add_subplot(3, 2, 6)
    axes.plot(time_index, angle[2], label='yaw (wind)', color='red')
    axes.plot(time_index, angle1[2], label='yaw (geen wind)', color='blue')
    axes.set_title('Hoeken')
    axes.set_xlabel('tijd (s)')
    axes.set_ylabel('hoek (deg)')
    axes.legend()

    plt.tight_layout(pad=0.4, w_pad=1, h_pad=1)
    plt.show()

    fig2 = plt.figure(figsize=(16, 8), dpi=100)
    ax = fig2.add_subplot(projection='3d')
    ax.plot(position[0], position[1], position[2], color='red', label='wind')
    ax.plot(position1[0], position1[1], position1[2], color='blue', label='geen wind')
    ax.set_title('Baan van drone wind vs geen wind')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.legend()
    plt.show()

total_plot()
