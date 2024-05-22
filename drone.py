import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import copy

# Functie voor cosinus in rotatiematrix
def c(x):
    return np.cos(x)

# Functie voor sinus in rotatiematrix
def s(x):
    return np.sin(x)

# Functie voor tangens in rotatiematrix
def t(x):
    return np.tan(x)


class Quadcopter():
    def __init__(self, pos, vel, angle, ang_vel, pos_ref, dt):
        # Constanten
        mass = 2.5 # Massa van quadcopter 2.5
        gravity = 9.81 # grav
        num_motors = 4 #Aantal motoren
        kt = 2e-7 # Torque constante
        
        # Initiele gedrag variablelen
        self.pos = pos # Positie vector [x, y, z]
        self.vel = vel # snelheid vector [vx, vy, vz]
        self.angle = angle # richting hoeken [roll, pitch, yaw]
        self.ang_vel = ang_vel # hoeksnelheid vector [p, q, r]
        self.lin_acc = np.array([0., 0., 0.]) # Lineare versnelling vector
        self.ang_acc = np.array([0., 0., 0.]) # heokversnelling vector
        
        # Reference states
        self.pos_ref = pos_ref # Gewenste positie
        self.vel_ref = [0., 0., 0.] # Gewenste snelheid
        self.lin_acc_ref = [0., 0., 0.] # gewenste lineare versnelling
        self.angle_ref = [0., 0., 0.] # gewenste hoek
        self.ang_vel_ref = [0., 0., 0.] # gewenste hoeksnelheid
        self.ang_acc_ref = [0., 0., 0.] # gewenste hoekversnelling
        
        # Tijd variabelen
        self.time = 0 # Simulatie tijd
        self.dt = dt # tijd stap
        
        # milieu
        self.gravity = gravity # grav
        self.density = 1.225 # luchtdichtheid
        
        
        # drone parameters, zelf gekozen op basis van andere bestaande modellen en berekend
        self.num_motors = num_motors # Number of motors
        self.mass = mass # Mass of the quadcopter
        self.Ixx = 8e-3 # Moment of inertia in x 8.11858e-5
        self.Iyy = 8e-3 # Moment of inertia in y 8.11858e-5
        self.Izz = 6e-3 # Moment of inertia in z 6.12233e-5
        self.A_ref = 0.02 # Oppervlakte voor drag 
        self.L = 0.2 # Lengte van body center tot propeller center
        self.kt = kt # Torque constante
        self.b_prop = 1e-9 # Proport constante voor motorsnelheid tot torque
        self.Cd = 1 # Drag coefficient
        self.thrust = mass * gravity # Totale thrust
        self.speeds = np.ones(num_motors) * ((mass * gravity) / (kt * num_motors)) # Initiele motor snelheden
        self.tau = np.zeros(3) # Torque vector
        
        # Limits
        self.maxT = 25 # Maximum thrust voor motor 20.7
        self.minT = .5 # Minimum thrust voor motor
        self.max_angle = math.pi/3 # Maximum toegestane hoek aan tijd stap
        
        # Moment of inertia 
        self.I = np.array([[self.Ixx, 0, 0],
                           [0, self.Iyy, 0],
                           [0, 0, self.Izz]])
        
        # Gravity vector
        self.g = np.array([0, 0, -gravity])
        
        # Simulation status
        self.done = False

    # Functie voor berekening van positie fout
    def calc_pos_error(self, pos):
        pos_error = self.pos_ref - pos
        return pos_error
    
    # Functie voor berekening snelheid fout
    def calc_vel_error(self, vel):
        vel_error = self.vel_ref - vel
        return vel_error

    # Functie voor berekening hoek fout
    def calc_ang_error(self, angle):
        angle_error = self.angle_ref - angle
        return angle_error

    # Functie voor berekening hoeksnelheid fout
    def calc_ang_vel_error(self, ang_vel):
        ang_vel_error = self.ang_vel_ref - ang_vel
        return ang_vel_error

    # Functie voor Euler van body-frame tot globael inertial frame
    def body2inertial_rotation(self):
        c1 = c(self.angle[0])
        s1 = s(self.angle[0])
        c2 = c(self.angle[1])
        s2 = s(self.angle[1])
        c3 = c(self.angle[2])
        s3 = s(self.angle[2])

        R = np.array([[c2*c3, c3*s1*s2 - c1*s3, s1*s3 + c1*s2*c3],
                      [c2*s3, c1*c3 + s1*s2*s3, c1*s3*s2 - c3*s1],
                      [-s2, c2*s1, c1*c2]])
    
        return R

    # Functie voor Euler rotatie van inertial tot body frame
    def inertial2body_rotation(self):
        R = np.transpose(self.body2inertial_rotation())
        return R

    # Functie om converteren hoeksnelheid van Euler dot tot inertial hoeksnelheid
    def thetadot2omega(self):
        R = np.array([[1, 0, -s(self.angle[1])],
                      [0, c(self.angle[0]), c(self.angle[1])*s(self.angle[0])],
                      [0, -s(self.angle[0]), c(self.angle[1])*c(self.angle[0])]])

        omega = np.matmul(R, self.ang_vel)
        return omega

    # Functie om converteren hoekversnelling van omega dot tot Euler dot
    def omegadot2Edot(self,omega_dot):
        R = np.array([[1, s(self.angle[0])*t(self.angle[1]), c(self.angle[0])*t(self.angle[1])],
                      [0, c(self.angle[0]), -s(self.angle[0])],
                      [0, s(self.angle[0])/c(self.angle[1]), c(self.angle[0])/c(self.angle[1])]])

        E_dot = np.matmul(R, omega_dot)
        self.ang_acc = E_dot

    # Functie om hoekversnelling te vinden in de inertial frame
    def find_omegadot(self, omega):
        omega = self.thetadot2omega() 
        omega_dot = np.linalg.inv(self.I).dot(self.tau - np.cross(omega, np.matmul(self.I, omega)))
        return omega_dot

    
    # Functie om lineare versnelling te vinden
    def find_lin_acc(self,wind):
        self.R_B2I = self.body2inertial_rotation()
        self.R_I2B = self.inertial2body_rotation()
    
        Thrust_body = np.array([0, 0, self.thrust])
        Thrust_inertial = np.matmul(self.R_B2I, Thrust_body)
    
        vel_bodyframe = np.matmul(self.R_I2B, self.vel)
        drag_body = -self.Cd * 0.5 * self.density * self.A_ref * (vel_bodyframe)**2
        drag_inertial = np.matmul(self.R_B2I, drag_body)
        weight = self.mass * self.g
        wind_force_x = 0.5*self.A_ref*self.density*(math.pow(wind[0],2))*math.cos(self.angle[1])
        wind_force_y = 0.5*self.A_ref*self.density*(math.pow(wind[1],2))*math.cos(self.angle[1])
        wind_force = [wind_force_x,wind_force_y,0]
        self.wind_inertial = np.matmul(self.R_B2I, wind_force)
        acc_inertial = (Thrust_inertial + drag_inertial + weight - self.wind_inertial) / self.mass
        self.lin_acc = acc_inertial

    # Functie om motor snelheden te berekenen om te voldoen aan gewenste thrust en torque 
    def des2speeds(self,thrust_des, tau_des):
        e1 = tau_des[0] * self.Ixx
        e2 = tau_des[1] * self.Iyy
        e3 = tau_des[2] * self.Izz
        n = self.num_motors
        weight_speed = thrust_des / (n*self.kt)
        motor_speeds = []
        motor_speeds.append(weight_speed - (e2/((n/2)*self.kt*self.L)) - (e3/(n*self.b_prop)))
        motor_speeds.append(weight_speed - (e1/((n/2)*self.kt*self.L)) + (e3/(n*self.b_prop)))
        motor_speeds.append(weight_speed + (e2/((n/2)*self.kt*self.L)) - (e3/(n*self.b_prop)))
        motor_speeds.append(weight_speed + (e1/((n/2)*self.kt*self.L)) + (e3/(n*self.b_prop)))

        thrust_all = np.array(motor_speeds) * (self.kt)
        over_max = np.argwhere(thrust_all > self.maxT)
        under_min = np.argwhere(thrust_all < self.minT)

        if over_max.size != 0:
            for i in range(over_max.size):
                motor_speeds[over_max[i][0]] = self.maxT / (self.kt)
        if under_min.size != 0:
            for i in range(under_min.size):
                motor_speeds[under_min[i][0]] = self.minT / (self.kt)
        
        self.speeds = motor_speeds

    # Functie om de torque van body te vinden
    def find_body_torque(self):
        tau = np.array([(self.L * self.kt * (self.speeds[3] - self.speeds[1])),
                        (self.L * self.kt * (self.speeds[2] - self.speeds[0])),
                        (self.b_prop * (-self.speeds[0] + self.speeds[1] - self.speeds[2] + self.speeds[3]))])

        self.tau = tau

    # Functie om staat van quadcopter up te daten voor elke tijdstap
    def step(self,wind):
        self.thrust = self.kt * np.sum(self.speeds)
        self.find_lin_acc(wind)
        self.find_body_torque()
        omega = self.thetadot2omega() 
        omega_dot = self.find_omegadot(omega)
        self.omegadot2Edot(omega_dot)
        self.ang_vel += self.dt * self.ang_acc
        self.angle += self.dt * self.ang_vel
    
        # wind toevoegen aan snelheid update 
        self.vel += self.dt * (self.lin_acc + self.wind_inertial )
    
        self.pos += self.dt * self.vel
        self.time += self.dt
