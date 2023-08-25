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


np.set_printoptions(precision=2)

#Orthogonal matrix
omat = np.array([[0,1],[-1,0]]) 

def solve(A,B,C):
    print("\nQ1.2.1")
    D = (B + C)/2
    E = (A + C)/2
    F = (A + B)/2
    print("D:", list(D))
    print("E:", list(E))
    print("F:", list(F))
#
    print("\nQ1.2.2")
    #Finding the equation of medians
    m4 = D - A
    m5 = E - B
    m6 = F - C
    n4 = omat@m4    #normal vector
    n5 = omat@m5    #normal vector
    n6 = omat@m6    #normal vector
    c4 = n4@A
    c5 = n5@B
    c6 = n6@C
    eqn4 = f"{n4}x = {c4}"
    eqn5 = f"{n5}x = {c5}"
    eqn6 = f"{n6}x = {c6}"
    print("The equation of line AD is",eqn4)
    print("The equation of line BE is",eqn5)
    print("The equation of line CF is",eqn6)
#
    print("\nQ1.2.3")
    def line_intersect(n1,A1,n2,A2):
	    N=np.block([[n1],[n2]])
	    p = np.zeros(2)
	    p[0] = n1@A1
	    p[1] = n2@A2
	    #Intersection
	    P=np.linalg.inv(N)@p
	    return P
    def norm_vec(A,B):
	    return np.matmul(omat, dir_vec(A,B))
    G=line_intersect(norm_vec(F,C),C,norm_vec(E,B),B)
    print("Point of intersection on BE and CF is: ",G)
#
    print("\nQ1.2.4")
    AG = np.linalg.norm(G - A)
    GD = np.linalg.norm(D - G)

    BG = np.linalg.norm(G - B)
    GE = np.linalg.norm(E - G)
 
    CG = np.linalg.norm(G - C)
    GF = np.linalg.norm(F - G)

    print("AG/GD= ",(AG/GD))
    print("BG/GE= ",(BG/GE))
    print("CG/GF= ",(CG/GF))
#
    print("\nQ1.2.5")
    Mat = np.array([[1,1,1],[A[0],D[0],G[0]],[A[1],D[1],G[1]]])
    rank = np.linalg.matrix_rank(Mat)
    if (rank==2):
	    print("As rank = 2, Hence proved that points A,G,D in a triangle are collinear")
    else:
	    print("Error")
#   
    print("\nQ1.2.6")
    print("G =", G) 
    G = (A + B + C) / 3
    print("(A+B+C)/3 = ", G)
    print("As both are equal, G is the centroid of triangle ABC")
#
    print("\nQ1.2.7")
    print(f"A - F = {A-F}")
    print(f"E - D = {E-D}")
    print("Hence, A-F = E-D")

def fig(A,B,C):
    D = (B + C)/2
    E = (A + C)/2
    F = (A + B)/2
    G = (A+B+C)/3
    #Generating all lines
    x_AB = line_gen(A,B)
    x_BC = line_gen(B,C)
    x_CA = line_gen(C,A)
    x_AD = line_gen(A,D)
    x_BE = line_gen(B,E)
    x_CF = line_gen(C,F)
    x_DF = line_gen(D,F)
    x_DE = line_gen(D,E)
    
    #Plotting all lines
    plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
    plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
    plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
    plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
    plt.plot(x_BE[0,:],x_BE[1,:],label='$BE$')
    plt.plot(x_CF[0,:],x_CF[1,:],label='$CF$')
    #Labeling the coordinates
    _A = A.reshape(-1,1)
    _B = B.reshape(-1,1)
    _C = C.reshape(-1,1)
    _D = D.reshape(-1,1)
    _E = E.reshape(-1,1)
    _F = F.reshape(-1,1)
    _G = G.reshape(-1,1)
    tri_coords = np.block([[_A, _B, _C, _D, _E, _F, _G]])
    plt.scatter(tri_coords[0, :], tri_coords[1, :])
    vert_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    for i, txt in enumerate(vert_labels):
        offset = 10
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
    plt.savefig('/home/yash/Desktop/geoPython/figs/2.png')
    plt.close()
    #Plotting all lines
    plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
    plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
    plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
    plt.plot(x_DF[0,:],x_DF[1,:],label='$DF$')
    plt.plot(x_DE[0,:],x_DE[1,:],label='$DE$')
    #Labeling the coordinates
    _A = A.reshape(-1,1)
    _B = B.reshape(-1,1)
    _C = C.reshape(-1,1)
    _D = D.reshape(-1,1)
    _E = E.reshape(-1,1)
    _F = F.reshape(-1,1)
    tri_coords = np.block([[_A, _B, _C, _D, _E, _F]])
    plt.scatter(tri_coords[0, :], tri_coords[1, :])
    vert_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, txt in enumerate(vert_labels):
        offset = 10
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
    plt.savefig('/home/yash/Desktop/geoPython/figs/2_7.png')
    plt.close()
