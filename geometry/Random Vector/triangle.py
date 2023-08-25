import numpy as np 
from codes import _1, _2, _3, _4, _5

#defining points A, B and C
#A = np.random.randint(-6,6,2)
#B = np.random.randint(-6,6,2)
#C = np.random.randint(-6,6,2)

A = np.array([4, -5])
B= np.array([-6, 2])
C = np.array([5, 4])

#printing points A, B and C
print("A = ",A)
print("B = ",B)
print("C = ",C)

_1.solve(A, B, C)
_1.fig(A,B,C)

_2.solve(A,B,C)
_2.fig(A,B,C)

_3.solve(A,B,C)
_3.fig(A,B,C)

_4.solve(A,B,C)
_4.fig(A,B,C)

_5.solve(A,B,C)
_5.fig(A,B,C)
