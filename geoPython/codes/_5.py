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


np.set_printoptions(precision=3)

#Orthogonal matrix
omat = np.array([[0,1],[-1,0]]) 

def dir_vec(A, B):
    return B - A

def norm_vec(A, B):
    return omat @ dir_vec(A, B)

def solve(A,B,C):
    print("\nQ1.5.1")
    def unit_vec(A,B):
	    return ((B-A)/np.linalg.norm(B-A))

    E_A = unit_vec(A,B) + unit_vec(A,C)
    F_A = np.array([E_A[1],(E_A[0]*(-1))])
    print("Internal Angular bisector of angle A is:",F_A,"*x = ",F_A@(A.T))
    E_B = unit_vec(A,B) + unit_vec(B,C)
    F_B = np.array([E_B[1],(E_B[0]*(-1))])
    print("Internal Angular bisector of angle B is:",F_B,"*x = ",F_B@(B.T))
    E_C = unit_vec(A,C) + unit_vec(B,C)
    F_C = np.array([E_C[1],(E_C[0]*(-1))])
    print("Internal Angular bisector of angle C is:",F_C,"*x = ",F_C@(C.T))

    k1=1
    k2=1

    p = np.zeros(2)
    t = norm_vec(B, C)
    n1 = t / np.linalg.norm(t)
    t = norm_vec(C, A)
    n2 = t / np.linalg.norm(t)
    t = norm_vec(A, B)
    n3 = t / np.linalg.norm(t)

    p[0] = n1 @ B - k1 * n2 @ C
    p[1] = n2 @ C - k2 * n3 @ A

    N = np.block([[n1 - k1 * n2],[ n2 - k2 * n3]])
    I = np.linalg.inv(N)@p
    r = n1 @ (B-I)

#
    print("\nQ1.5.2")
    I = np.linalg.inv(N)@p
    print("Coordinates of point I:", I)
#
    print("\nQ1.5.3")
    def angle_btw_vectors(v1, v2):
        dot_product = v1 @ v2
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        angle = np.arccos(dot_product / norm)
        angle_in_deg = np.degrees(angle)
        return angle_in_deg
    angle_BAI = angle_btw_vectors(B-A, I-A)
    angle_CAI = angle_btw_vectors(C-A, I-A)
    print("Angle BAI:", angle_BAI)
    print("Angle CAI:", angle_CAI)
#
    print("\nQ1.5.4")
    print("Distance from I to BC = ",r)
#
    print("\nQ1.5.5")
    r = n2 @ (A-I)
    print("Distance from I to AB = ",r)
    r = n3 @ (A-I)
    print("Distance from I to AC = ",r)
#
    print("\nQ1.5.7")
    print("Figure Generated")
#
    print("\nQ1.5.8")
    p=pow(np.linalg.norm(C-B),2)
    q=2*((C-B)@(I-B))
    r=pow(np.linalg.norm(I-B),2)-r*r

    Discre=q*q-4*p*r

    print("the Value of discriminant is ",abs(round(Discre,6)))
    k=((I-B)@(C-B))/((C-B)@(C-B))
    print("the value of parameter k is ",k)
    D3=B+k*(C-B)
    print("the point of tangency of incircle by side BC is ",D3)
    print("Hence we prove that side BC is tangent To incircle and also found the value of k!")
#
    print("\nQ1.5.9")
    k1=((I-A)@(A-B))/((A-B)@(A-B))
    k2=((I-A)@(A-C))/((A-C)@(A-C))
    #finding E_3 and F_3
    F3=A+(k1*(A-B))
    E3=A+(k2*(A-C))
    print("E3 = ",E3)
    print("F3 = ",F3)
#
    print("\nQ1.5.10")
    def norm(X,Y):
        magnitude=round(float(np.linalg.norm([X-Y])),3)
        return magnitude 
    print("AE_3=", norm(A,E3) ,"\nAF3=", norm(A,F3) ,"\nBD3=", norm(B,D3) ,"\nBF3=", norm(B,F3) ,"\nCD3=", norm(C,D3) ,"\nCE3=",norm(C,E3))
#
    print("\nQ1.5.11")
    a = np.linalg.norm(B-C)
    b = np.linalg.norm(C-A)
    c = np.linalg.norm(A-B)
    Y = np.array([[1,1,0],[0,1,1],[1,0,1]])
    X = np.linalg.solve(Y,[c,a,b])
    print("m = ", X[0])
    print("n = ", X[1])
    print("p = ", X[2])

def fig(A,B,C):
    k1=1
    k2=1

    p = np.zeros(2)
    t = norm_vec(B, C)
    n1 = t / np.linalg.norm(t)
    t = norm_vec(C, A)
    n2 = t / np.linalg.norm(t)
    t = norm_vec(A, B)
    n3 = t / np.linalg.norm(t)

    p[0] = n1 @ B - k1 * n2 @ C
    p[1] = n2 @ C - k2 * n3 @ A

    N = np.block([[n1 - k1 * n2],[ n2 - k2 * n3]])
    I = np.linalg.inv(N)@p
    r = n1 @ (B-I)
    k=((I-B)@(C-B))/((C-B)@(C-B))
    D3=B+k*(C-B)
    k1=((I-A)@(A-B))/((A-B)@(A-B))
    k2=((I-A)@(A-C))/((A-C)@(A-C))
    F3=A+(k1*(A-B))
    E3=A+(k2*(A-C))

    #Generating all lines
    x_AB = line_gen(A,B)
    x_BC = line_gen(B,C)
    x_CA = line_gen(C,A)
    x_AI = line_gen(A,I)
    x_BI = line_gen(B,I)
    x_CI = line_gen(C,I)
    #Plotting all lines
    plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
    plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
    plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
    plt.plot(x_AI[0,:],x_AI[1,:],label='$AI$')
    plt.plot(x_BI[0,:],x_BI[1,:],label='$BI$')
    plt.plot(x_CI[0,:],x_CI[1,:],label='$CI$')
    #generating the incircle
    x_icirc= circ_gen(I,r)
    plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')
    #Labeling the coordinates
    _A = A.reshape(-1,1)
    _B = B.reshape(-1,1)
    _C = C.reshape(-1,1)
    _I = I.reshape(-1,1)
    _D3 = D3.reshape(-1,1)
    _E3 = E3.reshape(-1,1)
    _F3 = F3.reshape(-1,1)
    tri_coords = np.block([[_A, _B, _C, _I,_D3, _E3, _F3]])
    plt.scatter(tri_coords[0, :], tri_coords[1, :])
    vert_labels = ['A', 'B', 'C', 'I', 'D3', 'E3', 'F3']
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
    plt.savefig('/home/yash/Desktop/geoPython/figs/5.png')
    plt.close()
