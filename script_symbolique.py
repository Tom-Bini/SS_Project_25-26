import sympy as sp

#----------------------------------------QUESTION 2.2----------------------------------------
#Position:
x, y, phi, x_ref, y_ref, phi_ref, k_x, k_y, k_phi = sp.symbols("x y phi x_ref y_ref phi_ref k_x k_y k_phi")
l, m, inertia, g = sp.symbols("l m inertia g")

u_s = m * (g + k_y * (y_ref - y))
u_d = 2 * inertia / l * k_phi * (- 1/g * k_x * (x_ref - x) - phi)

#----------------------------------------QUESTION 2.3----------------------------------------

matrix_A = sp.zeros(6, 6)

matrix_A[0,3] = 1
matrix_A[1,4] = 1
matrix_A[2,5] = 1

matrix_A[3,2] = - g

matrix_B = sp.zeros(6, 2)

matrix_B[4,0] = 1 / m
matrix_B[5,1] = l / (2 * inertia)

matrix_C = sp.zeros(2, 6)

matrix_C[0,0] = 1
matrix_C[1,1] = 1

matrix_D = sp.zeros(2,2)

matrix_K = sp.zeros(2,6)

matrix_K[0,1] = m * k_y
matrix_K[1,0] = -2 * inertia * k_phi * k_x / (g * l)
matrix_K[1,2] = 2 * inertia * k_phi / l

matrix_Kr = sp.zeros(2, 2)

matrix_Kr[0,1] = m * k_y
matrix_Kr[1,0] = -2 * inertia * k_phi * k_x / (g * l)

matrix_A_tilde = matrix_A - (matrix_B * matrix_K)

matrix_B_tilde = matrix_B * matrix_Kr
matrix_C_tilde = matrix_C - matrix_D * matrix_K
matrix_D_tilde = matrix_D * matrix_Kr

#----------------------------------------QUESTION 2.4----------------------------------------

matrix_A_tilde_sp = sp.Matrix(matrix_A_tilde)

eigenvalues_matrix_A_tilde = matrix_A_tilde_sp.eigenvals()


#----------------------------------------QUESTION 3.1----------------------------------------

s = sp.Symbol("s")

print("Calcul de H(s) = C_tilde * (sI - A_tilde)^{-1} * B_tilde + D_tilde")

sI = sp.zeros(6, 6)

sI[0, 0] = s
sI[1, 1] = s
sI[2, 2] = s
sI[3, 3] = s
sI[4, 4] = s
sI[5, 5] = s

sI_A = sI - matrix_A_tilde

sI_A_inv = sI_A.inv()

H_s = matrix_C_tilde * sI_A_inv * matrix_B_tilde + matrix_D_tilde

sp.pprint(H_s)