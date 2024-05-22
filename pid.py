import numpy as np
import math

class PID_Controller():

    def __init__(self, Kp, Kd, Ki, Ki_sat, dt):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.Ki_sat = Ki_sat
        self.dt = dt
        
        # Totale som van fouten voor x,y,z or roll,pitch,yaw
        self.int = [0., 0., 0.]

    def control_update(self, pos_error, vel_error):
        
        #Totale fout update
        self.int += pos_error * self.dt
        over_mag = np.argwhere(np.array(self.int) > np.array(self.Ki_sat))
        if over_mag.size != 0:
            for i in range(over_mag.size):
                mag = abs(self.int[over_mag[i][0]]) #magnitude krijgen om richting te vinden
                self.int[over_mag[i][0]] = (self.int[over_mag[i][0]] / mag) * self.Ki_sat[over_mag[i][0]] #behouden richting (sign) maar limiet saturatie 

        
        #Gewenste versnelling output
        des_acc = self.Kp * pos_error + self.Ki * self.int + self.Kd * vel_error
        return des_acc
