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
    print("\nQ1.3.1")
    print("Normal vector of AD : ", B-C)
#
    print("\nQ1.3.2")
    N1 = B-C
    print("Equation of AD is ",N1,"X = ", N1@A)
#
    print("\nQ1.3.3")
    N2 = A-C
    print("Equation of BE is ",N2,"X = ", N2@B)
    N3 = A-B
    print("Equation of CF is ",N3,"X = ", N3@C)
#
    print("\nQ1.3.4")
    Mat = [[N2[0],N2[1]],[N3[0],N3[1]]]
    H = np.linalg.solve(Mat,[N2@B,N3@C])
    print("Point of intersection of BE and CF is : ", H)
#
    print("\nQ1.3.5")
    result = int(((A - H).T) @ (B - C))
    print("Ans = ",result)
    if result == 0:
        print("(A - H)^T (B - C) = 0\tHence Verified")
    else:
        print("(A - H)^T (B - C)) != 0\tHence the given statement is wrong")

def fig(A,B,C):
    #Intersection of two lines
    def line_intersect(n1,A1,n2,A2):
        N=np.vstack((n1,n2))
        p = np.zeros(2)
        p[0] = n1@A1
        p[1] = n2@A2
        #Intersection
        P=np.linalg.inv(N)@p
        return P
    
    #Foot of the Altitude
    def alt_foot(A,B,C):
        m = B-C
        n = np.matmul(omat,m) 
        N=np.vstack((m,n))
        p = np.zeros(2)
        p[0] = m@A 
        p[1] = n@B
        #Intersection
        P=np.linalg.inv(N.T)@p
        return P

    D = alt_foot(A,B,C)
    E = alt_foot(B,A,C)
    F = alt_foot(C,A,B)
    
    #Finding orthocentre
    H = line_intersect(norm_vec(B,E),E,norm_vec(C,F),F)
    
    #Generating all lines
    x_AB = line_gen(A,B)
    x_BC = line_gen(B,C)
    x_CA = line_gen(C,A)
    x_AD = line_gen(A,D)
    x_BE = line_gen(B,E)
    x_CF = line_gen(C,F)
    x_AH = line_gen(A,H)
    x_BH = line_gen(B,H)
    x_CH = line_gen(C,H)
    x_AE = line_gen(A,E)
    x_AF = line_gen(A,F)
    #Plotting all lines
    #Plotting all lines
    plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
    plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
    plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
    plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
    plt.plot(x_BE[0,:],x_BE[1,:],label='$BE_1$')
    plt.plot(x_AE[0,:],x_AE[1,:],linestyle = 'dashed',label='$AE_1$')
    plt.plot(x_CF[0,:],x_CF[1,:],label='$CF_1$')
    plt.plot(x_AF[0,:],x_AF[1,:],linestyle = 'dashed',label='$AF_1$')
    plt.plot(x_CH[0,:],x_CH[1,:],label='$CH$')
    plt.plot(x_BH[0,:],x_BH[1,:],label='$BH$')
    plt.plot(x_AH[0,:],x_AH[1,:],linestyle = 'dashed',label='$AH$')

    #Labeling the coordinates
    _A = A.reshape(-1,1)
    _B = B.reshape(-1,1)
    _C = C.reshape(-1,1)
    _D = D.reshape(-1,1)
    _E = E.reshape(-1,1)
    _F = F.reshape(-1,1)
    _H = H.reshape(-1,1)
    tri_coords = np.block([[_A, _B, _C, _D, _E, _F, _H]])
    plt.scatter(tri_coords[0, :], tri_coords[1, :])
    vert_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'H']
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
    plt.savefig('/home/yash/Desktop/geoPython/figs/3.png')
    plt.close()
