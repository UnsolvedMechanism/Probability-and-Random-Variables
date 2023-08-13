import numpy as np
import matplotlib.pyplot as pl

#defining vertices of triangle in matrix format
A = np.array([1,-1])
B = np.array([-4,6])
C = np.array([-3,-5])

#plotting A B and C on graph
Xpoints = np.array([1,-4,-3])
Ypoints = np.array([-1,6,-5])

pl.plot(Xpoints,Ypoints,'o')
pl.grid()
pl.xlabel('x-axis')
pl.ylabel('y-axis')
pl.text(A[0],A[1],"  A")
pl.text(B[0],B[1],"  B")
pl.text(C[0],C[1],"  C")

pl.savefig('../figs/fig.png')

pl.show()

#creating array to check rank:
M = np.array([[1,1,1],[1,-4,3],[1,6,-5]])

#find rank of matrix
r = np.linalg.matrix_rank(M)

#solving the equations
#X = np.linalg.solve(Y,[AB_i,AC_i])

#printing output 
print(r)
