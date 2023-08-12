import numpy as np

#defining equations of perpendicular bisectors of AB and AC
AB_coeff = np.array([5,-7])
AB_i = -25
AC_coeff = np.array([4,4])
AC_i = -16

#creating array containing coefficients
Y = np.array([AB_coeff,AC_coeff])

#solving the equations
X = np.linalg.solve(Y,[AB_i,AC_i])

#printing output 
print(X)
