from math import sqrt
import numpy as np

#defining vertices of triangle in matrix format
A = np.array([1,-1])
B = np.array([-4,6])
C = np.array([-3,-5])

#finding sidelengths a, b & c
a = np.absolute(sqrt((B-C)@(B-C)))
b = np.absolute(sqrt((A-C)@(A-C)))
c = np.absolute(sqrt(B-A)@(B-A))

s = (a+b+c)/2

m = s-a
n = s-b
p = s-c

print("m = ",m)
print("n = ",n)
print("p = ",p)