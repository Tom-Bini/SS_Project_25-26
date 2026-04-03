import numpy as np
from scipy import integrate
import visualization as vz

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
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Magie de la Team Pekka 1", FPS)
    
def case_2():
    u_s = m * g
    u_d = 0.01
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Magie de la Team Pekka 2", FPS)

def case_3():
    u_s = m * g
    u_d = 0.1
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Magie de la Team Pekka 3", FPS)
    
def case_conclusion():
    x0 = np.array([0, 0, np.pi/180, 0, 0, 0])
    u_s = m * g
    u_d = 0
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Magie de la Team Pekka Conclus", FPS)

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
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Magie de la Brigade du Kiff 1", FPS)
    
def linear_case_2():
    u_s = m * g
    u_d = 0.01
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(linear_system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Magie de la Brigade du Kiff 2", FPS)

def linear_case_3():
    u_s = m * g
    u_d = 0.1
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(linear_system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Magie de la Brigade du Kiff 3", FPS)


def case_1m():
    u_s = m * g + 0.223
    u_d = 0
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Magie de la Team Pekka 1M", FPS)
    
case_1m()