import numpy as np 
import math
import sys                                          #for path to external scripts
import matplotlib.pyplot as plt
import matplotlib.image as mping

sys.path.insert(0, '/home/yash/Desktop/geoPython/CoordGeo')        
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
import subprocess
import shlex

def solve(A, B, C):
    print("\nQ1.1.1:")
    AB = B - A
    BC = C - B
    CA = A - C
    print("The direction vector of AB is ",AB)
    print("The direction vector of BC is ",BC)
    print("The direction vector of CA is ",CA)
#
    print("\nQ1.1.2:")
    print("The length of AB is:",np.linalg.norm(AB))
    print("The length of BC is:",np.linalg.norm(BC))
    print("The length of CA is:",np.linalg.norm(CA))
#
    print("\nQ1.1.3:")
    Mat = np.array([[1,1,1],[A[0],B[0],C[0]],[A[1],B[1],C[1]]])
    rank = np.linalg.matrix_rank(Mat)
    if(rank==2):
    	print("Hence proved that points A,B,C are collinear as rank = 2")
    else:
    	print("The given points are not collinear as rank = ", rank)
#
    print("\nQ1.1.4:")
    print("Parametric of AB form is x=",A,"+ k",AB)
    print("Parametric of BC form is x=",B,"+ k",BC)
    print("Parametric of CA form is x=",C,"+ k",CA)
#
    print("\nQ1.1.5:")
    #Orthogonal matrix
    omat = np.array([[0,1],[-1,0]])

    def norm_vec(C,B):
        return omat@dir_vec(C,B)
    
    n=norm_vec(A,B)
    pro=n@A
    print("Normal form of equation of AB: ",n,"x=",pro)
    n=norm_vec(C,B)
    pro=n@B
    print("Normal form of equation of BC : ",n,"x=",pro)
    n=norm_vec(C,A)
    pro=n@C
    print("Normal form of equation of CA : ",n,"x=",pro)
#
    print("\nQ1.1.6:")
    cross_product = np.cross(AB,CA)
    magnitude = np.linalg.norm(cross_product)
    area = 0.5 * magnitude
    print("Area of triangle ABC:", area)
#
    print("\nQ1.1.7:")
    dotA=((B-A).T)@(C-A)
    NormA=(np.linalg.norm(B-A))*(np.linalg.norm(C-A))
    dotB=(A-B).T@(C-B)
    NormB=(np.linalg.norm(A-B))*(np.linalg.norm(C-B))
    dotC=(A-C).T@(B-C)
    NormC=(np.linalg.norm(A-C))*(np.linalg.norm(B-C))
    print('value of angle A: ', np.degrees(np.arccos((dotA)/NormA)))
    print('value of angle B: ', np.degrees(np.arccos((dotB)/NormB)))
    print('value of angle C: ', np.degrees(np.arccos((dotC)/NormC)))

def fig(A,B,C):
    #Generating all lines
    x_AB = line_gen(A,B)
    x_BC = line_gen(B,C)
    x_CA = line_gen(C,A)
    #Plotting all lines
    plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
    plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
    plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
    #Labeling the coordinates
    _A = A.reshape(-1,1)
    _B = B.reshape(-1,1)
    _C = C.reshape(-1,1)
    tri_coords = np.block([[_A, _B, _C]])
    plt.scatter(tri_coords[0, :], tri_coords[1, :])
    vert_labels = ['A', 'B', 'C']
    for i, txt in enumerate(vert_labels):
        offset = 10 if txt == 'C' else -10
        plt.annotate(txt,
                     (tri_coords[0, i], tri_coords[1, i]),
                     textcoords="offset points",
                     xytext=(0, offset),
                     ha='center')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(loc='best')
    plt.grid()
    plt.axis('equal')
    plt.savefig('/home/yash/Desktop/geoPython/figs/1.png')
    plt.close()
