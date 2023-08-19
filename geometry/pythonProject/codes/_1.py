import numpy as np 
import math
import sys                                          #for path to external scripts
import matplotlib.pyplot as plt
import matplotlib.image as mping

sys.path.insert(0, 'geometry/pythonProject')        #path to my scripts

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
import subprocess
import shlex

def helloWorld():
    print("Hello World!")

AB = np.array([1,1])
BC = np.array([1,1])
CA = np.array([1,1])

def _1_1(A, B, C):
    print("Q1.1.1:")
    AB = B - A
    BC = C - B
    CA = A - C
    print("The direction vector of AB is ",AB)
    print("The direction vector of BC is ",BC)
    print("The direction vector of CA is ",CA)
    print()

def _1_2(A, B, C):
    print("Q1.1.2:")
    AB = B - A
    BC = C - B
    CA = A - C
    _AB = AB.reshape(-1,1)
    print("The length of AB is:",math.sqrt(AB@_AB))

    _BC = BC.reshape(-1,1)
    print("The length of BC is:",math.sqrt(BC@_BC))

    _CA = CA.reshape(-1,1)
    print("The length of AC is:",math.sqrt(CA@_CA))
    print()

def _1_3(A, B, C):
    print("Q1.1.3:")
    Mat = np.array([[1,1,1],[A[0],B[0],C[0]],[A[1],B[1],C[1]]])
    rank = np.linalg.matrix_rank(Mat)
    if(rank==2):
    	print("Hence proved that points A,B,C are collinear as rank = 2")
    else:
    	print("The given points are not collinear as rank = ", rank)
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
    plt.savefig('/home/yash/Desktop/Probability Git Clone/Probability-and-Random-Variables/geometry/pythonProject/figs/_1_3.png')
    # plt.show()

def _1_4(A,B,C):
    m1=(B-A)
    m2=(C-B)
    m3=(A-C)
    print("Q1.1.4:")
    print("\nParametric of AB form is x:",A,"+ k",m1)
    print("Parametric of BC form is x:",B,"+ k",m2)
    print("Parametric of CA form is x:",C,"+ k",m3,"\n")

def _1_5_n(A,B,C):
    
    def dir_vec(C,B):
        return C-B

    def norm_vec(C,B):
        return omat@dir_vec(C,B)

    n=norm_vec(C,B)
    pro=n@B
    print(n,"x=",pro)
    def line_gen(A,B):
        len =10
        dim = A.shape[0]
        x_AB = np.zeros((dim,len))
        lam_1 = np.linspace(0,1,len)
        for i in range(len):
            temp1 = A + lam_1[i]*(B-A)
            x_AB[:,i]= temp1.T
        return x_AB
    x_BC = line_gen(B,C)
    plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
    B = B.reshape(-1,1)
    C = C.reshape(-1,1)
    tri_coords = np.block([[B,C]])
    plt.scatter(tri_coords[0,:], tri_coords[1,:])
    vert_labels = ['B','C']
    for i, txt in enumerate(vert_labels):
        plt.annotate(txt, # this is the text
                (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center

def _1_5(A,B,C):
    _1_5_n(A,B,C)
    _1_5_n(B,C,A)
    _1_5_n(C,A,B)
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
    plt.savefig('/home/yash/Desktop/Probability Git Clone/Probability-and-Random-Variables/geometry/pythonProject/figs/_1_5.png')

def _1_6(A,B,C):
    AB = A - B
    AC = A - C
    cross_product = np.cross(AB,AC)
    magnitude = np.linalg.norm(cross_product)
    area = 0.5 * magnitude
    print("\nQ1.1.6:")
    print("Area of triangle ABC:", area)

def _1_7(A,B,C):
    dotA=((B-A).T)@(C-A)
    NormA=(np.linalg.norm(B-A))*(np.linalg.norm(C-A))

    dotB=(A-B).T@(C-B)
    NormB=(np.linalg.norm(A-B))*(np.linalg.norm(C-B))

    dotC=(A-C).T@(B-C)
    NormC=(np.linalg.norm(A-C))*(np.linalg.norm(B-C))
    print('value of angle A: ', np.degrees(np.arccos((dotA)/NormA)))
    print('value of angle B: ', np.degrees(np.arccos((dotB)/NormB)))
    print('value of angle C: ', np.degrees(np.arccos((dotC)/NormC)))
