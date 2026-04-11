import numpy as np
from scipy import integrate
import visualization as vz
import matplotlib.pyplot as plt
import sympy as sp

#----------------------------------------QUESTION 1.6----------------------------------------

NUM_STEP = 575
FPS = 144

t_start = 0
t_end = 3 #secondes, la durée de simulation

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

    matrix_A[3,2] = - u_s / m
    
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
    u_s = m * g + 0.1
    u_d = 0
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(linear_system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Système linéaire, Cas 1", FPS)
    
def linear_case_2():
    u_s = m * g
    u_d = 0.01
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(linear_system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Système linéaire, Cas 2", FPS)

def linear_case_3():
    u_s = m * g
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
    u_s = m * g + 0.223
    u_d = 0
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Système non-linéaire, 1m vertical", FPS)
    
def systemInputsVariables(t, state_variables, u_s, u_d):
    step = min(int(t / ((t_end - t_start) / NUM_STEP)), NUM_STEP - 1)
    
    x, y, phi, x_dot, y_dot, phi_dot = state_variables
    
    x_ddot = - u_s[step] / m * np.sin(phi)
    
    y_ddot = u_s[step] / m * np.cos(phi) - g
    
    phi_ddot = l / (2 * inertia) * u_d[step]
    
    return [x_dot, y_dot, phi_dot, x_ddot, y_ddot, phi_ddot]
    
def case_noisy():
    u_s = m * g + 0.223
    u_d = 0
    
    mean = 0
    std = 1

    u_s_array = np.random.normal(mean, std, NUM_STEP) + u_s
    u_d_array = np.random.normal(mean, std, NUM_STEP) + u_d
    
    sol = integrate.solve_ivp(systemInputsVariables, t_span, x0, t_eval = t_eval, args = (u_s_array, u_d_array))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Système non-linéaire avec bruit gaussien", FPS)
    
#----------------------------------------QUESTION 2.5----------------------------------------

r = 1.0 #m
omega = 0.2 #rad/s
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
t_end = 3 #secondes, la durée de simulation

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

#Fréquences dominantes
freq_1 = 3
freq_2 = 10
freq_3 = 25

amplitude_1 = 0.060
amplitude_2 = 0.032
amplitude_3 = 0.0320

def artificial_noise(t):
    s1 = amplitude_1 * np.cos(2 * np.pi * freq_1 * t)  # Pic à 3 Hz
    s2 = amplitude_2 * np.cos(2 * np.pi * freq_2 * t)  # Pic à 10 Hz
    s3 = amplitude_3 * np.cos(2 * np.pi * freq_3 * t)  # Pic à 25 Hz
    
    return s1 + s2 + s3

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
    k_phi = 44
    
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