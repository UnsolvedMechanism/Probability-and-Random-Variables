import numpy as np

#creating array to check rank:
M = np.array([[1,1,1],[1,-4,3],[1,6,-5]])

#find rank of matrix
r = np.linalg.matrix_rank(M)

#solving the equations
#X = np.linalg.solve(Y,[AB_i,AC_i])

#printing output 
print(r)
