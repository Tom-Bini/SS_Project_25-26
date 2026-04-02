import numpy as np
from scipy import integrate
import visualization as vz

NUM_STEP = 100000

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

def cas_1():
    u_s = 0.1
    u_d = 0
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Magie de la Team Pekka 1", 144)
    
def cas_2():
    u_s = 0
    u_d = 0.01
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Magie de la Team Pekka 2", 144)

def cas_3():
    u_s = 0
    u_d = 0.1
    u_s_array = np.full(NUM_STEP, u_s)
    u_d_array = np.full(NUM_STEP, u_d)
    
    sol = integrate.solve_ivp(system, t_span, x0, t_eval = t_eval, args = (u_s, u_d))
    vz.animate(sol.t, sol.y[0], sol.y[1], sol.y[2], u_s_array, u_d_array, "Magie de la Team Pekka 3", 144)

cas_1()
cas_2()
cas_3()