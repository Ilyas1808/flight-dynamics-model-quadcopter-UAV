import numpy as np
import matplotlib.pyplot as plt


U_ref = 10  # windsnelheid (m/s)
sigma = 0.1  # intensiteit turbulentie
L = 100  # length scale turbulentie (m)
A = 1  # Spectral amplitude parameter
N = 1024  # Aantal frequentie components
delta_t = 0.1  # Tijd stap (s)
T = 100  # Duur simulatie (s)
delta_x = 10  # Spatiale stap (m)

# frequenties en omega
frequencies = np.fft.fftfreq(N, d=delta_t)
omega = 2 * np.pi * frequencies

# amplitudes and fases berekeningen
amplitudes = np.sqrt(2 * A / ((L ** (-2) + omega ** 2) ** (5/6))) * sigma
phases = np.random.uniform(0, 2 * np.pi, N)  #random maken

#Tijd series voor initiele punten maken
time = np.arange(0, T, delta_t)
u_t = np.zeros_like(time)

for i, t in enumerate(time):
    u_t[i] = np.sum(amplitudes * np.cos(omega * t + phases))

# tijd series normaliseren
u_t /= np.max(np.abs(u_t))

# correlatie functie (von Kármán model)
def von_karman_coherence(delta_x, L):
    return np.exp(-delta_x / L)

# Simulatie stromingsveld volgens baan
def simulate_flow_path(U_ref, sigma, L, A, N, delta_t, T, delta_x):
    path_length = int(T * U_ref / delta_x)
    flow_field = np.zeros((path_length, len(time)))
    for i in range(path_length):
        x = i * delta_x
        coherence = von_karman_coherence(x, L)
        correlated_amplitudes = amplitudes * coherence
        for j, t in enumerate(time):
            flow_field[i, j] = np.sum(correlated_amplitudes * np.cos(omega * t + phases))
    return flow_field

#Genereren van stromingsveld volgens baan 
flow_field = simulate_flow_path(U_ref, sigma, L, A, N, delta_t, T, delta_x)

# Downloaden als csv
np.savetxt('simulated_flow_field_hlilkjln.csv', flow_field, delimiter=',')

#plotten
plt.figure(figsize=(10, 6))
for i in range(flow_field.shape[0]):
    plt.plot(time, flow_field[i, :], alpha=0.3)
plt.xlabel('Tijd (s)')
plt.ylabel('Gesimuleerde windsnelheid (m/s)')
plt.title('Realistische simulatie windsnelheid in stad')
plt.legend()
plt.show()
