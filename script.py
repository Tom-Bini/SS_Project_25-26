import numpy as np
from scipy import integrate
import visualization as vz
import matplotlib.pyplot as plt

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

    matrix_A[3,2] = - u_s / m * np.cos(phi)
    matrix_A[4,2] = - u_s / m * np.sin(phi)
    
    matrix_B = np.zeros((6, 2))
    
    matrix_B[3,0] = - np.sin(phi) / m
    matrix_B[4,0] = np.cos(phi) / m
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
    
#----------------------------------------QUESTION 2.2----------------------------------------
#Position:
x = None
y = None
phi = None

#Position de référence:
x_ref = None
y_ref = None
phi_ref = None

#Gains (> 0):
k_x = None
k_y = None
k_phi = None

u_s = m * (g + k_y * (y_ref - y))
phi_ref = - 1/g * k_x * (x_ref - x)
u_d = 2 * inertia / l * k_phi * (phi_ref - phi)

#----------------------------------------QUESTION 2.3----------------------------------------

matrix_A = np.zeros((6, 6))

matrix_A[0,3] = 1
matrix_A[1,4] = 1
matrix_A[2,5] = 1

matrix_A[3,2] = - u_s / m * np.cos(phi)
matrix_A[4,2] = - u_s / m * np.sin(phi)

matrix_B = np.zeros((6, 2))

matrix_B[3,0] = - np.sin(phi) / m
matrix_B[4,0] = np.cos(phi) / m
matrix_B[5,1] = l / (2 * inertia)

matrix_C = np.zeros((2, 6))

matrix_C[0,0] = 1
matrix_C[1,1] = 1

matrix_D = np.zeros((2,2))

matrix_K = np.zeros((2, 6))

matrix_K[0][1] = m * k_y
matrix_K[1][0] = -2 * inertia * k_phi * k_x / (g * l)
matrix_K[1][2] = 2 * inertia * k_phi / l

matrix_Kr = np.zeros((2, 2))

matrix_Kr[0][1] = m * k_y
matrix_Kr[1][0] = -2 * inertia * k_phi * k_x / (g * l)

matrix_A_tilde = matrix_A - matrix_B @ matrix_K
matrix_B_tilde = matrix_B @ matrix_Kr
matrix_C_tilde = matrix_C - matrix_D @ matrix_K
matrix_D_tilde = matrix_D @ matrix_Kr