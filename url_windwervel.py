import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from drone import Quadcopter
from pid import PID_Controller
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


######################################
# data = """
# validdate;wind_speed_10m:ms;wind_gusts_10m_24h:ms;wind_dir_10m:d
# 00:00;4.0;0;0
# 1:00;4.0;10.0;82.8
# 2:00;4.1;10.0;83.7
# 3:00;5.5;10.0;76.6
# 4:00;5.9;10.7;67.7
# 5:00;5.8;10.7;63.0
# 6:00;5.7;11.0;62.4
# """


# Datum en tijd van nu
now = datetime.now()

#  + 6u tov nu
end_time = now + timedelta(hours=6)

# omzetten
start_time_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

data=""
url = f"https://api.meteomatics.com/{start_time_str}--{end_time_str}:PT1H/wind_speed_10m:ms,wind_gusts_10m_24h:ms,wind_dir_10m:d/50.4714388,4.1852956/html"

username = '...' #confidential, make an account 
password = '...'

response = requests.get(url, auth=(username, password))

# Control if the request is succesfull
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    csv_data = soup.find('pre', id='csv').text.strip()

    
    lines = csv_data.split("\n")[1:]  
    edited_lines = []

    for line in lines:
        parts = line.split(";")
        date_time = parts[0]
        time_only = date_time.split("T")[1][:5]  
        
        if time_only.startswith("00:"):
            edited_time = "0:00"
        else:
            edited_time = time_only.lstrip("0")
        new_line = f"{edited_time};{parts[1]};{parts[2]};{parts[3]}"
        edited_lines.append(new_line)

   
    if edited_lines[0].startswith("0:00"):
        edited_lines = edited_lines[1:]

 
    edited_csv_data = "\n".join(edited_lines)
    print(edited_csv_data)
else:
    print("Error:", response.status_code)


lines = edited_csv_data.split('\n')
hours = [entry.split(';')[0] for entry in lines]
bearing = [float(entry.split(';')[3]) for entry in lines]
wind_speed = [float(entry.split(';')[1]) for entry in lines]
wind_gusts = [float(entry.split(';')[2]) for entry in lines]
wind_noise = [-5,5]
# omzetten naar int
hours_int = [int(hour.split(':')[0]) for hour in hours]

# Windsnelheid en gust 
time = np.linspace(0, 1, len(hours))  # tijdsinterval
wind_speed_curve = np.array(wind_speed)
wind_gusts_curve = np.array(wind_gusts)

sim_start = 0
number_of_hours = len(hours_int)
print("NUMBER OF HOURS",number_of_hours)
sim_end = number_of_hours*60
print("SIMULATION time",sim_end)

# cosinus curve
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

mid_range = len(interpolated_wind)/2
for i in range(0,len(interpolated_wind),1):  #7
    if i<mid_range:
        rand = 0
    if i>mid_range and i<len(interpolated_wind)*0.75:
        rand = random.randint(-1, 1)
    if i>len(interpolated_wind)*0.75:
        rand = random.randint(-5, 5)
    interpolated_wind[i] = interpolated_wind[i] + rand
    
angle = np.array(angle)
interpolated_wind = np.array(interpolated_wind)
true_bearing = np.array(true_bearing)


print("LENGTH OF TIME",len(t))
# Plot
plt.figure(figsize=(10.6,4))
plt.plot(np.linspace(0, len(hours_int), len(interpolated_wind)), interpolated_wind, label='Windsnelheid (m/s)')
plt.xlabel('Tijd (uren)')
plt.ylabel('Windsnelheid (m/s)')
plt.title('Geïnterpoleerde windsnelheid over 7 uur (via URL)')
plt.legend()
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
target_position = np.array([[10., 20., 70],[20., 50., 10],[60., 40., 20],[80., 10., 90]])
#ref=np.array([60., 40., 20])
pos = [0., 0., 0.]
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


print("Interpolated Bearing",len(interpolated_wind),len(true_bearing),len(time_index),time_index[-1 ])

time_index = np.array(time_index)
location_change_time = np.array_split(time_index,len(target_position))
count = 0
for ref in target_position:
    quadcopter = Quadcopter(pos, vel, ang, ang_vel, ref, dt)
    pos_controller = PID_Controller(Kp_pos, Kd_pos, Ki_pos, Ki_sat_pos, dt)
    angle_controller = PID_Controller(Kp_ang, Kd_ang, Ki_ang, Ki_sat_ang, dt)

    for time in enumerate(location_change_time[count]):
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
    #    print("TESTING",mag_acc, des_acc[1] ,quadcopter.angle[1],des_acc[0])
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
    count+=1
    pos = ref
    vel = np.array([quadcopter.vel[0],quadcopter.vel[1],quadcopter.vel[2]])
    ang = np.array([quadcopter.angle[0],quadcopter.angle[1],quadcopter.angle[2]])
def total_plot():
    fig = plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')

    axes = fig.add_subplot(3, 2, 1)
    axes.plot(time_index, position[0], label='x')
    axes.plot(time_index, position[1], label='y')
    axes.set_title('Postions')
    axes.set_xlabel('time (min)')
    axes.set_ylabel('position (m)')
    axes.legend()

    axes = fig.add_subplot(3, 2, 2)
    axes.plot(time_index, position[2], label='z')
    axes.set_title('Vertical Position')
    axes.set_xlabel('time (min)')
    axes.set_ylabel('altitude (m)')

    axes = fig.add_subplot(3, 2, 3)
    axes.plot(time_index, velocity[0], label='x_dot')
    axes.plot(time_index, velocity[1], label='y_dot')
    axes.plot(time_index, velocity[2], label='z_dot')
    axes.set_title('Linear Velocity')
    axes.set_xlabel('time (min)')
    axes.set_ylabel('velocity (m/min)')
    axes.legend()

    axes = fig.add_subplot(3, 2, 4)
    axes.plot(time_index, angle[0], label='roll')
    axes.plot(time_index, angle[1], label='pitch')
    axes.plot(time_index, angle[2], label='yaw')
    axes.set_title('Angles')
    axes.set_xlabel('time (min)')
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
