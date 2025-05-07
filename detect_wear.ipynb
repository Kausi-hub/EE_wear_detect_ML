import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, measured_value):
        error = setpoint - measured_value
        self.integral += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

def clarke_transform(ia, ib, ic):
    """ Converts 3-phase currents (ia, ib, ic) to 2-phase (alpha, beta) """
    alpha = (2/3) * (ia - 0.5*ib - 0.5*ic)
    beta = (2/3) * ((np.sqrt(3)/2) * (ib - ic))
    return alpha, beta

def park_transform(alpha, beta, theta):
    """ Converts (alpha, beta) to (d, q) reference frame using rotor angle theta """
    d = alpha * np.cos(theta) + beta * np.sin(theta)
    q = -alpha * np.sin(theta) + beta * np.cos(theta)
    return d, q

def inverse_park_transform(vd, vq, theta):
    """ Converts (d, q) voltages back to (alpha, beta) frame """
    alpha = vd * np.cos(theta) - vq * np.sin(theta)
    beta = vd * np.sin(theta) + vq * np.cos(theta)
    return alpha, beta

def inverse_clarke_transform(alpha, beta):
    """ Converts (alpha, beta) back to 3-phase voltages (va, vb, vc) """
    va = alpha
    vb = -0.5 * alpha + (np.sqrt(3)/2) * beta
    vc = -0.5 * alpha - (np.sqrt(3)/2) * beta
    return va, vb, vc

# Simulation Parameters
num_steps = 100
theta = np.linspace(0, 2*np.pi, num_steps)  # Rotor position (electrical angle)
target_torque = 5  # Target torque (related to iq)

# Initialize PID Controllers
id_pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
iq_pid = PIDController(kp=1.5, ki=0.1, kd=0.05)

# Motor State Variables
id_measured, iq_measured = 0, 0
id_ref = 0  # We usually set Id = 0 for maximum torque efficiency
iq_ref = target_torque  # Target torque control

# Data Logging
id_list, iq_list, va_list, vb_list, vc_list = [], [], [], [], []

for t in range(num_steps):
    # Clarke Transform: Assume balanced phase currents
    ia, ib, ic = np.sin(theta[t]), np.sin(theta[t] - 2*np.pi/3), np.sin(theta[t] + 2*np.pi/3)
    alpha, beta = clarke_transform(ia, ib, ic)

    # Park Transform: Get dq currents
    d, q = park_transform(alpha, beta, theta[t])

    # PID Controllers to regulate dq currents
    vd = id_pid.compute(id_ref, d)  # Regulate Id (Flux control)
    vq = iq_pid.compute(iq_ref, q)  # Regulate Iq (Torque control)

    # Inverse Park Transform: Convert back to alpha-beta voltages
    alpha_v, beta_v = inverse_park_transform(vd, vq, theta[t])

    # Inverse Clarke Transform: Convert to 3-phase voltages
    va, vb, vc = inverse_clarke_transform(alpha_v, beta_v)

    # Store data for visualization
    id_list.append(d)
    iq_list.append(q)
    va_list.append(va)
    vb_list.append(vb)
    vc_list.append(vc)

# Plot Results
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(id_list, label="Id (Flux)")
plt.plot(iq_list, label="Iq (Torque)", linestyle="dashed")
plt.axhline(y=iq_ref, color='r', linestyle='--', label="Iq Ref (Torque)")
plt.title("DQ Current Control")
plt.xlabel("Time Step")
plt.ylabel("Current (A)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(va_list, label="Va")
plt.plot(vb_list, label="Vb", linestyle="dashed")
plt.plot(vc_list, label="Vc", linestyle="dotted")
plt.title("3-Phase Voltages (Va, Vb, Vc)")
plt.xlabel("Time Step")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(theta, label="Rotor Electrical Angle")
plt.title("Rotor Angle")
plt.xlabel("Time Step")
plt.ylabel("Angle (rad)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
