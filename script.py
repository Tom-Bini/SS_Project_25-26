import numpy as np
from scipy import integrate
import visualization as vz
import matplotlib.pyplot as plt
import sympy as sp
from scipy.interpolate import interp1d

#----------------------------------------QUESTION 1.6----------------------------------------

NUM_STEP = 575
FPS = 144

t_start = 0
t_end = 3

m = 1
l = 0.3
inertia = 0.02
g = 9.81

t_span = (t_start, t_end)
t_eval = np.linspace(t_start, t_end, NUM_STEP)

x0 = np.array([0, 0, 0, 0, 0, 0])

def system(t, state_variables, u_s, u_d):
    
    x, y, phi, x_dot, y_dot, phi_dot = state_variables
    
    x_ddot = - u_s / m * np.sin(phi)
    
    y_ddot = u_s / m * np.cos(phi) - g
    
    phi_ddot = l / (2 * inertia) * u_d
    
    return [x_dot, y_dot, phi_dot, x_ddot, y_ddot, phi_ddot]

def case_1():
    u_s = m * g + 0.1
    u_d = 0
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Système non-linéaire, Cas 1", FPS)

def case_2():
    u_s = m * g
    u_d = 0.01
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Système non-linéaire, Cas 2", FPS)

def case_3():
    u_s = m * g
    u_d = 0.1
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Système non-linéaire, Cas 3", FPS)


def linear_system(t, state_variables, u_s, u_d):
    x, y, phi, x_dot, y_dot, phi_dot = state_variables
    
    matrix_state = np.array([x, y, phi, x_dot, y_dot, phi_dot])
    
    matrix_input = np.array([u_s, u_d])
    
    matrix_A = np.zeros((6, 6))

    matrix_A[0,3] = 1
    matrix_A[1,4] = 1
    matrix_A[2,5] = 1

    matrix_A[3,2] = - g
    
    matrix_B = np.zeros((6, 2))
    
    matrix_B[4,0] = 1 / m
    matrix_B[5,1] = l / (2 * inertia)
    
    matrix_C = np.zeros((2, 6))
    
    matrix_C[0,0] = 1
    matrix_C[1,1] = 1
    
    matrix_D = np.zeros((2,2))
    
    dxdt = matrix_A @ matrix_state + matrix_B @ matrix_input
    
    return dxdt

def linear_case_1():
    u_s = 0.1
    u_d = 0
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(linear_system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Système linéaire, Cas 1", FPS)
    
def linear_case_2():
    u_s = 0
    u_d = 0.01
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(linear_system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Système linéaire, Cas 2", FPS)

def linear_case_3():
    u_s = 0
    u_d = 0.1
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(linear_system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Système linéaire, Cas 3", FPS)

#----------------------------------------QUESTION 1.7----------------------------------------

def case_conclusion():
    x0 = np.array([0, 0, np.pi/180, 0, 0, 0])
    u_s = m * g
    u_d = 0
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Système non-linéaire, test stabilité", FPS)
    
#----------------------------------------QUESTION 2.1----------------------------------------

def case_1m():
    def u_s_func(t):
        if t < 1.0:
            return m * g + 1.0
        elif t < 2.0:
            return m * g - 1.0
        else:
            return m * g
            
    def u_d_func(t):
        return 0.0

    def system_wrapper(t, y):
        current_us = u_s_func(t)
        current_ud = u_d_func(t)
        return system(t, y, current_us, current_ud)

    t_array = np.linspace(0, t_span[1], NUM_STEP)
    u_s_array = np.array([u_s_func(t) for t in t_array])
    u_d_array = np.array([u_d_func(t) for t in t_array])
    
    sol = integrate.solve_ivp(system_wrapper, t_span, x0, t_eval=t_eval)
    
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Système non-linéaire, 1m vertical", FPS)

def systemInputsVariables(t, state_variables, u_s, u_d):
    step = min(int(t / ((t_end - t_start) / NUM_STEP)), NUM_STEP - 1)
    
    x, y, phi, x_dot, y_dot, phi_dot = state_variables
    
    x_ddot = - u_s[step] / m * np.sin(phi)
    
    y_ddot = u_s[step] / m * np.cos(phi) - g
    
    phi_ddot = l / (2 * inertia) * u_d[step]
    
    return [x_dot, y_dot, phi_dot, x_ddot, y_ddot, phi_ddot]
    
def case_1m_noisy():
    mean = 0.0
    std = np.sqrt(2) * 0.05

    def u_s_base(t):
        if t < 1.0:
            return m * g + 1.0
        elif t < 2.0:
            return m * g - 1.0
        else:
            return m * g
            
    def u_d_base(t):
        return 0.0

    t_array = np.linspace(0, t_span[1], NUM_STEP)
    
    u_s_clean = np.array([u_s_base(t) for t in t_array])
    u_d_clean = np.array([u_d_base(t) for t in t_array])
    
    u_s_noisy_array = u_s_clean + np.random.normal(mean, std, NUM_STEP)
    u_d_noisy_array = u_d_clean + np.random.normal(mean, std, NUM_STEP)
    
    u_s_interp = interp1d(t_array, u_s_noisy_array, bounds_error=False, fill_value="extrapolate")
    u_d_interp = interp1d(t_array, u_d_noisy_array, bounds_error=False, fill_value="extrapolate")

    def system_wrapper(t, y):
        current_us = u_s_interp(t)
        current_ud = u_d_interp(t)
        return system(t, y, current_us, current_ud) 

    sol = integrate.solve_ivp(system_wrapper, t_span, x0, t_eval=t_eval)
 
    u_s_sol_array = u_s_interp(sol.t)
    u_d_sol_array = u_d_interp(sol.t)
    
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_sol_array, u_d_sol_array, "1m vertical avec bruit", FPS)
    
#----------------------------------------QUESTION 2.5----------------------------------------

r = 1.0
omega = 0.2
x0 = np.array([1, 1.5, 0, 0, 0, 0])

def feedbacked_system(t, state_variables, k_x, k_y, k_phi, mean, std):
    x, y, phi, x_dot, y_dot, phi_dot = state_variables
    
    x_ref = r * np.cos(omega * t)
    y_ref = r * np.sin(omega * t) + 1.5
    
    u_s = m * (g + k_y * (y_ref - y))
    u_d = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x) - phi)
    
    seed = int(t * 1000)
   
    u_s_noisy = u_s + np.random.default_rng(seed).normal(mean, std)
    u_d_noisy = u_d + np.random.default_rng(seed + 1).normal(mean, std)

    x_ddot = - u_s_noisy / m * np.sin(phi)
    
    y_ddot = u_s_noisy / m * np.cos(phi) - g
    
    phi_ddot = l / (2 * inertia) * u_d_noisy
    
    return [x_dot, y_dot, phi_dot, x_ddot, y_ddot, phi_ddot]

def q2_5_case1():
    k_x = 3
    k_y = 3
    k_phi = 20
    
    mean = 0
    std = 0
    
    t_start = 0
    t_end = 10 * np.pi
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, NUM_STEP)
    
    sol = integrate.solve_ivp(feedbacked_system, t_span, x0, t_eval = t_eval, args = (k_x, k_y, k_phi, mean, std))
    
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]
    t_sol = sol.t
    
    x_ref = r * np.cos(omega * t_sol)
    y_ref = r * np.sin(omega * t_sol) + 1.5
    
    u_s_array = m * (g + k_y * (y_ref - y_sol))
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    for i in range(len(t_sol)):
        seed = int(t_sol[i] * 1000)
        
        u_s_array[i] += np.random.default_rng(seed).normal(mean, std)
        u_d_array[i] += np.random.default_rng(seed + 1).normal(mean, std)
    
    vz.animate(t_sol, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q2.5 Cas 1", FPS)
    
    for i in range(NUM_STEP):
        square_sum = 0
        square_sum += (x_sol[i] - x_ref[i])**2 + (y_sol[i] - y_ref[i])**2
    
    rmse = np.sqrt(1/NUM_STEP * square_sum)
    print(rmse)

    
def q2_5_case2():
    k_x = 3
    k_y = 3
    k_phi = 20
    
    mean = 0
    std = 0.01
    
    t_span = 0
    t_end = 10 * np.pi
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, NUM_STEP)
    
    sol = integrate.solve_ivp(feedbacked_system, t_span, x0, t_eval = t_eval, args = (k_x, k_y, k_phi, mean, std))
    
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]
    t_sol = sol.t
    
    x_ref = r * np.cos(omega * t_sol)
    y_ref = r * np.sin(omega * t_sol) + 1.5
    
    u_s_array = m * (g + k_y * (y_ref - y_sol))
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    for i in range(len(t_sol)):
        seed = int(t_sol[i] * 1000)
        
        u_s_array[i] += np.random.default_rng(seed).normal(mean, std)
        u_d_array[i] += np.random.default_rng(seed + 1).normal(mean, std)
    
    vz.animate(t_sol, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q2.5 Cas 2", FPS)
    
    for i in range(NUM_STEP):
        square_sum = 0
        square_sum += (x_sol[i] - x_ref[i])**2 + (y_sol[i] - y_ref[i])**2
    
    rmse = np.sqrt(1/NUM_STEP * square_sum)
    print(rmse)
    
def q2_5_case3():
    k_x = 3
    k_y = 3
    k_phi = 20
    
    mean = 0
    std = 0.05
    
    t_span = 0
    t_end = 10 * np.pi
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, NUM_STEP)
    
    sol = integrate.solve_ivp(feedbacked_system, t_span, x0, t_eval = t_eval, args = (k_x, k_y, k_phi, mean, std))
    
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]
    t_sol = sol.t
    
    x_ref = r * np.cos(omega * t_sol)
    y_ref = r * np.sin(omega * t_sol) + 1.5
    
    u_s_array = m * (g + k_y * (y_ref - y_sol))
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    for i in range(len(t_sol)):
        seed = int(t_sol[i] * 1000)
        
        u_s_array[i] += np.random.default_rng(seed).normal(mean, std)
        u_d_array[i] += np.random.default_rng(seed + 1).normal(mean, std)
    
    vz.animate(t_sol, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q2.5 Cas 3", FPS)
    
    for i in range(NUM_STEP):
        square_sum = 0
        square_sum += (x_sol[i] - x_ref[i])**2 + (y_sol[i] - y_ref[i])**2
    
    rmse = np.sqrt(1/NUM_STEP * square_sum)
    print(rmse)
    
def q2_5_case4():
    k_x = 10
    k_y = 10
    k_phi = 40
    
    mean = 0
    std = 0
    
    t_span = 0
    t_end = 10 * np.pi
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, NUM_STEP)
    
    sol = integrate.solve_ivp(feedbacked_system, t_span, x0, t_eval = t_eval, args = (k_x, k_y, k_phi, mean, std))
    
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]
    t_sol = sol.t
    
    x_ref = r * np.cos(omega * t_sol)
    y_ref = r * np.sin(omega * t_sol) + 1.5
    
    u_s_array = m * (g + k_y * (y_ref - y_sol))
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    for i in range(len(t_sol)):
        seed = int(t_sol[i] * 1000)
        
        u_s_array[i] += np.random.default_rng(seed).normal(mean, std)
        u_d_array[i] += np.random.default_rng(seed + 1).normal(mean, std)
    
    vz.animate(t_sol, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q2.5 Cas 4", FPS)
    
    for i in range(NUM_STEP):
        square_sum = 0
        square_sum += (x_sol[i] - x_ref[i])**2 + (y_sol[i] - y_ref[i])**2
    
    rmse = np.sqrt(1/NUM_STEP * square_sum)
    print(rmse)
    
def q2_5_case5():
    k_x = 10
    k_y = 10
    k_phi = 40
    
    mean = 0
    std = 0.01
    
    t_span = 0
    t_end = 10 * np.pi
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, NUM_STEP)
    
    sol = integrate.solve_ivp(feedbacked_system, t_span, x0, t_eval = t_eval, args = (k_x, k_y, k_phi, mean, std))
    
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]
    t_sol = sol.t
    
    x_ref = r * np.cos(omega * t_sol)
    y_ref = r * np.sin(omega * t_sol) + 1.5
    
    u_s_array = m * (g + k_y * (y_ref - y_sol))
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    for i in range(len(t_sol)):
        seed = int(t_sol[i] * 1000)
        
        u_s_array[i] += np.random.default_rng(seed).normal(mean, std)
        u_d_array[i] += np.random.default_rng(seed + 1).normal(mean, std)
    
    vz.animate(t_sol, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q2.5 Cas 5", FPS)
    
    for i in range(NUM_STEP):
        square_sum = 0
        square_sum += (x_sol[i] - x_ref[i])**2 + (y_sol[i] - y_ref[i])**2
    
    rmse = np.sqrt(1/NUM_STEP * square_sum)
    print(rmse)
    
def q2_5_case6():
    k_x = 10
    k_y = 10
    k_phi = 40
    
    mean = 0
    std = 0.05
    
    t_span = 0
    t_end = 10 * np.pi
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, NUM_STEP)
    
    sol = integrate.solve_ivp(feedbacked_system, t_span, x0, t_eval = t_eval, args = (k_x, k_y, k_phi, mean, std))
    
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]
    t_sol = sol.t
    
    x_ref = r * np.cos(omega * t_sol)
    y_ref = r * np.sin(omega * t_sol) + 1.5
    
    u_s_array = m * (g + k_y * (y_ref - y_sol))
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    for i in range(len(t_sol)):
        seed = int(t_sol[i] * 1000)
        
        u_s_array[i] += np.random.default_rng(seed).normal(mean, std)
        u_d_array[i] += np.random.default_rng(seed + 1).normal(mean, std)
    
    vz.animate(t_sol, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q2.5 Cas 6", FPS)
    
    for i in range(NUM_STEP):
        square_sum = 0
        square_sum += (x_sol[i] - x_ref[i])**2 + (y_sol[i] - y_ref[i])**2
    
    rmse = np.sqrt(1/NUM_STEP * square_sum)
    print(rmse)

#----------------------------------------QUESTION 2.6----------------------------------------

t_start = 0
t_end = 3

t_span = (t_start, t_end)
t_eval = np.linspace(t_start, t_end, NUM_STEP)

x0 = np.array([0, 0, 0, 0, 0, 0])

def q2_6():

    t, u = np.loadtxt("mesure_poussee.csv", delimiter=",", skiprows=1).T
    freqs = np.fft.rfftfreq(len(t), d=t[1]-t[0])
    ampl = (2.0 / len(t)) * np.abs(np.fft.rfft(u - 1))
    plt.plot(freqs, ampl)
    plt.xlabel("Fréquence (s$^{-1}$)")
    plt.ylabel("Force (N)")
    plt.grid(alpha = 0.3)
    plt.grid(which = "minor", alpha = 0.2, linestyle = ":")
    plt.minorticks_on()
    plt.savefig("q2.6.png")
        
#----------------------------------------QUESTION 2.7----------------------------------------


freq_1 = 3
freq_2 = 10
freq_3 = 25

amplitude_1 = 0.061
amplitude_2 = 0.030
amplitude_3 = 0.011

def artificial_noise(t):
    s1 = amplitude_1 * np.cos(2 * np.pi * freq_1 * t)  # Pic à 3 Hz
    s2 = amplitude_2 * np.cos(2 * np.pi * freq_2 * t)  # Pic à 10 Hz
    s3 = amplitude_3 * np.cos(2 * np.pi * freq_3 * t)  # Pic à 25 Hz
    
    return s1 + s2 + s3


def q2_7_comparison():
    t, u = np.loadtxt("mesure_poussee.csv", delimiter=",", skiprows=1).T

    bruit_reel = u - 1.0 
    bruit_modele = [artificial_noise(ti) for ti in t]
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, bruit_reel, label="Bruit Réel (Expérimental)", alpha=0.5)
    plt.plot(t, bruit_modele, label="Bruit Modèle (Série de Fourier)", color='red', linewidth=2)
    
    plt.xlim(0, 0.5)
    plt.xlabel("Temps (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("comparaison_temporelle_bruit.png")

    plt.figure(figsize=(10, 5))
    freqs = np.fft.rfftfreq(len(t), d=t[1]-t[0])
    
    ampl_reel = (2.0 / len(t)) * np.abs(np.fft.rfft(bruit_reel))
    ampl_modele = (2.0 / len(t)) * np.abs(np.fft.rfft(bruit_modele))
    
    plt.plot(freqs, ampl_reel, label="Spectre Réel", alpha=0.5)
    plt.stem(freqs, ampl_modele, linefmt='r-', markerfmt='ro', basefmt=" ", label="Pics Modèle (Q7)")
    
    plt.xlim(0, 40)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig("comparaison_frequentielle_bruit.png")

#----------------------------------------QUESTION 2.8----------------------------------------

x0 = np.array([1, 1.5, 0, 0, 0, 0])

def feedbacked_noisy_system(t, state_variables, k_x, k_y, k_phi):
    x, y, phi, x_dot, y_dot, phi_dot = state_variables
    
    x_ref = r * np.cos(omega * t)
    y_ref = r * np.sin(omega * t) + 1.5
    
    u_s = m * (g + k_y * (y_ref - y))
    u_d = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x) - phi)
   
    u_s_noisy = u_s + 2 * artificial_noise(t)
    u_d_noisy = u_d 

    x_ddot = - u_s_noisy / m * np.sin(phi)
    
    y_ddot = u_s_noisy / m * np.cos(phi) - g
    
    phi_ddot = l / (2 * inertia) * u_d_noisy
    
    return [x_dot, y_dot, phi_dot, x_ddot, y_ddot, phi_ddot]

def q2_8():
    k_x = 11
    k_y = 11
    k_phi = 67
    
    t_start = 0
    t_end = 10 * np.pi
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, NUM_STEP)
    
    sol = integrate.solve_ivp(feedbacked_noisy_system, t_span, x0, t_eval = t_eval, args = (k_x, k_y, k_phi))
    
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]
    t_sol = sol.t
    
    x_ref = r * np.cos(omega * t_sol)
    y_ref = r * np.sin(omega * t_sol) + 1.5
    
    u_s_array = m * (g + k_y * (y_ref - y_sol))
    
    for i in range(len(u_s_array)):
        t = t_eval[i]
        u_s_array[i] += 2 * artificial_noise(t)
        
    
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    for i in range(NUM_STEP):
        square_sum = 0
        square_sum += (x_sol[i] - x_ref[i])**2 + (y_sol[i] - y_ref[i])**2
    
    rmse = np.sqrt(1/NUM_STEP * square_sum)
    print(rmse)
    vz.animate(t_sol, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q2.8", FPS)


#----------------------------------------QUESTION 3.3----------------------------------------

x0 = np.array([1, 1.5, 0, 0, 0, 0])


def q3_3_case1():
    k_x = 5
    k_y = 5
    k_crit = 20
    k_phi = 0.99 * k_crit
    
    mean = 0
    std = 0
    
    t_start = 0
    t_end = 30
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, NUM_STEP)
    
    sol = integrate.solve_ivp(feedbacked_system, t_span, x0, t_eval = t_eval, args = (k_x, k_y, k_phi, mean, std))
    
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]
    t_sol = sol.t
    
    x_ref = r * np.cos(omega * t_sol)
    y_ref = r * np.sin(omega * t_sol) + 1.5
    
    u_s_array = m * (g + k_y * (y_ref - y_sol))
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    for i in range(len(t_sol)):
        seed = int(t_sol[i] * 1000)
        
        u_s_array[i] += np.random.default_rng(seed).normal(mean, std)
        u_d_array[i] += np.random.default_rng(seed + 1).normal(mean, std)
    
    vz.animate(t_sol, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q3.3 Cas 1", FPS)
    
    for i in range(NUM_STEP):
        square_sum = 0
        square_sum += (x_sol[i] - x_ref[i])**2 + (y_sol[i] - y_ref[i])**2
    
    rmse = np.sqrt(1/NUM_STEP * square_sum)
    print(rmse)



def q3_3_case2():
    k_x = 5
    k_y = 5
    k_crit = 20
    k_phi = 1.01 * k_crit
    
    mean = 0
    std = 0
    
    t_start = 0
    t_end = 30
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, NUM_STEP)
    
    sol = integrate.solve_ivp(feedbacked_system, t_span, x0, t_eval = t_eval, args = (k_x, k_y, k_phi, mean, std))
    
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]
    t_sol = sol.t
    
    x_ref = r * np.cos(omega * t_sol)
    y_ref = r * np.sin(omega * t_sol) + 1.5
    
    u_s_array = m * (g + k_y * (y_ref - y_sol))
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    for i in range(len(t_sol)):
        seed = int(t_sol[i] * 1000)
        
        u_s_array[i] += np.random.default_rng(seed).normal(mean, std)
        u_d_array[i] += np.random.default_rng(seed + 1).normal(mean, std)
    
    vz.animate(t_sol, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q3.3 Cas 2", FPS)
    
    for i in range(NUM_STEP):
        square_sum = 0
        square_sum += (x_sol[i] - x_ref[i])**2 + (y_sol[i] - y_ref[i])**2
    
    rmse = np.sqrt(1/NUM_STEP * square_sum)
    print(rmse)


#----------------------------------------QUESTION 3.4----------------------------------------

def bode_phase(num_coeffs, den_coeffs, omega):

    poles = np.roots(den_coeffs)
    zeros = np.roots(num_coeffs)
    
    phase = np.zeros_like(omega, dtype=float)

    wn_poles = [np.imag(p) for p in poles if np.isclose(np.real(p), 0) and np.imag(p) > 0]
    wn_zeros = [np.imag(z) for z in zeros if np.isclose(np.real(z), 0) and np.imag(z) > 0]

    for wn in wn_poles:
        phase[omega > wn] -= 180.0

    for wn in wn_zeros:
        phase[omega > wn] += 180.0
        
    return phase

def bode():
    kx = 3
    ky = 3
    kphi = 20

    omega = np.logspace(-2, 2, 10000)
    s = 1j * omega

    H11_complex = (kphi * kx) / (s**4 + kphi * s**2 + kphi * kx)
    num_H11 = [kphi * kx]
    den_H11 = [1, 0, kphi, 0, kphi * kx]


    H22_complex = ky / (s**2 + ky)
    num_H22 = [ky]
    den_H22 = [1, 0, ky]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # --- Tracé H11 ---
    axes[0, 0].semilogx(omega, 20 * np.log10(np.abs(H11_complex)))
    axes[0, 0].set_title("Bode amplitude — H11 (xref→x)")
    axes[0, 0].set_ylabel("Amplitude (dB)")
    axes[0, 0].grid(True, which='both', alpha=0.3)

    # Calcul explicite de la phase via les pôles
    phase_H11 = bode_phase(num_H11, den_H11, omega)
    axes[1, 0].semilogx(omega, phase_H11)
    axes[1, 0].set_title("Bode phase explicite — H11 (xref→x)")
    axes[1, 0].set_xlabel("ω (rad/s)")
    axes[1, 0].set_ylabel("Phase (°)")
    axes[1, 0].grid(True, which='both', alpha=0.3)

    # --- Tracé H22 ---
    axes[0, 1].semilogx(omega, 20 * np.log10(np.abs(H22_complex)))
    axes[0, 1].set_title("Bode amplitude — H22 (yref→y)")
    axes[0, 1].grid(True, which='both', alpha=0.3)

    # Calcul explicite de la phase via les pôles
    phase_H22 = bode_phase(num_H22, den_H22, omega)
    axes[1, 1].semilogx(omega, phase_H22)
    axes[1, 1].set_title("Bode phase explicite — H22 (yref→y)")
    axes[1, 1].set_xlabel("ω (rad/s)")
    axes[1, 1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig("bode_poles_explicite.png")


#----------------------------------------QUESTION 3.5----------------------------------------

mean = 0
std = 0
r = 1.0
k_x = 3
k_y = 3
k_phi = 20
omega_n = np.sqrt(k_y)

def feedbacked_system_2(t, state_variables, k_x, k_y, k_phi, mean, std):
    x, y, phi, x_dot, y_dot, phi_dot = state_variables
    
    x_ref = 0
    y_ref = np.sin(omega * t)
    
    u_s = m * (g + k_y * (y_ref - y))
    u_d = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x) - phi)
    
    seed = int(t * 1000)
   
    u_s_noisy = u_s + np.random.default_rng(seed).normal(mean, std)
    u_d_noisy = u_d + np.random.default_rng(seed + 1).normal(mean, std)

    x_ddot = - u_s_noisy / m * np.sin(phi)
    
    y_ddot = u_s_noisy / m * np.cos(phi) - g
    
    phi_ddot = l / (2 * inertia) * u_d_noisy
    
    return [x_dot, y_dot, phi_dot, x_ddot, y_ddot, phi_ddot]



def q3_5_omega_0_1():
    global omega
    omega = 0.1 * omega_n
    
    t_end = 4 * (2 * np.pi / omega)
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, NUM_STEP)
    
    x0 = np.array([0, 0, 0, 0, (k_y * omega) / (k_y - omega ** 2), 0])
    
    sol = integrate.solve_ivp(feedbacked_system_2, t_span, x0, t_eval=t_eval, args=(k_x, k_y, k_phi, mean, std))

    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]

    x_ref = np.zeros_like(sol.t)
    y_ref = np.sin(omega * sol.t)

    u_s_array = m * (g + k_y * (y_ref - y_sol))
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    vz.animate(sol.t, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q3.5 - Omega = 0.1 Wn", FPS)

def q3_5_omega_0_5():
    global omega
    omega = 0.5 * omega_n
    
    t_end = 4 * (2 * np.pi / omega)
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, NUM_STEP)
    
    x0 = np.array([0, 0, 0, 0, (k_y * omega) / (k_y - omega ** 2), 0])
    
    sol = integrate.solve_ivp(feedbacked_system_2, t_span, x0, t_eval=t_eval, args=(k_x, k_y, k_phi, mean, std))

    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]

    x_ref = np.zeros_like(sol.t)
    y_ref = np.sin(omega * sol.t)

    u_s_array = m * (g + k_y * (y_ref - y_sol))
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    vz.animate(sol.t, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q3.5 - Omega = 0.5 Wn", FPS)

def q3_5_omega_1_5():
    global omega
    omega = 1.5 * omega_n
    
    t_end = 4 * (2 * np.pi / omega)
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, NUM_STEP)
    
    x0 = np.array([0, 0, 0, 0, (k_y * omega) / (k_y - omega ** 2), 0])
    
    sol = integrate.solve_ivp(feedbacked_system_2, t_span, x0, t_eval=t_eval, args=(k_x, k_y, k_phi, mean, std))
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]

    x_ref = np.zeros_like(sol.t)
    y_ref = np.sin(omega * sol.t)

    u_s_array = m * (g + k_y * (y_ref - y_sol))
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    vz.animate(sol.t, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q3.5 - Omega = 1.5 Wn", FPS)

def q3_5_omega_3_0():
    global omega
    omega = 3.0 * omega_n
    
    t_end = 4 * (2 * np.pi / omega)
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, NUM_STEP)
    
    x0 = np.array([0, 0, 0, 0, (k_y * omega) / (k_y - omega ** 2), 0])
    
    sol = integrate.solve_ivp(feedbacked_system_2, t_span, x0, t_eval=t_eval, args=(k_x, k_y, k_phi, mean, std))
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]

    x_ref = np.zeros_like(sol.t)
    y_ref = np.sin(omega * sol.t)

    u_s_array = m * (g + k_y * (y_ref - y_sol))
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    vz.animate(sol.t, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q3.5 - Omega = 3.0 Wn", FPS)


def q3_5_omega_cutoff():
    global omega
    omega = omega_n
    
    t_end = 4 * (2 * np.pi / omega)
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, NUM_STEP)
    
    x0 = np.array([0, 0, 0, 0, 0, 0])
    
    sol = integrate.solve_ivp(feedbacked_system_2, t_span, x0, t_eval=t_eval, args=(k_x, k_y, k_phi, mean, std))
    x_sol = sol.y[0]
    y_sol = sol.y[1]
    phi_sol = sol.y[2]

    x_ref = np.zeros_like(sol.t)
    y_ref = np.sin(omega * sol.t)

    u_s_array = m * (g + k_y * (y_ref - y_sol))
    u_d_array = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x_sol) - phi_sol)
    
    vz.animate(sol.t, x_sol, y_sol, phi_sol, u_s_array, u_d_array, "Q3.5 - Omega = Wn", FPS)


q3_5_omega_cutoff()