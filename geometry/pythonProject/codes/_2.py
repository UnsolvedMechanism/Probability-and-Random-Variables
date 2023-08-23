import sys                                          #for path to external scripts
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, 'geometry/pythonProject')        #path to my scripts
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

import subprocess
import shlex

def _1(A,B,C):
    D = (B + C)/2
    E = (A + C)/2
    F = (A + B)/2
    print("\nQ1.2.1")
    print("D:", list(D))
    print("E:", list(E))
    print("F:", list(F))

    x_AB = line_gen(A,B)
    x_BC = line_gen(B,C)
    x_CA = line_gen(C,A)

    #Plotting all lines
    plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
    plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
    plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

    #Labeling the coordinates
    A = A.reshape(-1,1)
    B = B.reshape(-1,1)
    C = C.reshape(-1,1)
    D = D.reshape(-1,1)
    E = E.reshape(-1,1)
    F = F.reshape(-1,1)
    tri_coords = np.block([[A,B,C,D,E,F]])
    plt.scatter(tri_coords[0,:], tri_coords[1,:])
    vert_labels = ['A','B','C','D','E','F']
    for i, txt in enumerate(vert_labels):
        plt.annotate(txt, # this is the text
                     (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend([])
    plt.grid() # minor
    plt.axis('equal')
    plt.savefig('/home/yash/Desktop/Probability Git Clone/Probability-and-Random-Variables/geometry/pythonProject/figs/_2_1.png')
