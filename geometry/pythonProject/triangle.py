import numpy as np 
import math
import sys
from codes import _1

sys.path.insert(0, '/home/yash/Desktop/Python Project/CoordGeo')        #path to my scripts
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
import subprocess
import shlex

simlen = 2
#defining points A, B and C
A = np.random.randint(-6,6,2)
B = np.random.randint(-6,6,2)
C = np.random.randint(-6,6,2)

#printing points A, B and C
print("A = ",A)
print("B = ",B)
print("C = ",C)
print()

#_1.helloWorld()

_1._1_1(A,B,C)

_1._1_2(A,B,C)

_1._1_3(A,B,C)

_1._1_4(A,B,C)

_1._1_5(A,B,C)

_1._1_6(A,B,C)

_1._1_7(A,B,C)
