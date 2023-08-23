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
    print("\nQ1.4.1")
    def perpendicular_bisector(B, C):
        midBC=(B+C)/2
        dir=B-C
        constant = -dir.T @ midBC
        return dir,constant
    equation_coeff1,const1 = perpendicular_bisector(A, B)
    equation_coeff2,const2 = perpendicular_bisector(B, C)
    equation_coeff3,const3 = perpendicular_bisector(C, A)
    print(f'Equation for perpendicular bisector of AB: ({equation_coeff1[0]:.2f})x + ({equation_coeff1[1]:.2f})y + ({const1:.2f}) = 0')
    print(f'Equation for perpendicular bisector of  BC: ({equation_coeff2[0]:.2f})x + ({equation_coeff2[1]:.2f})y + ({const2:.2f}) = 0')
    print(f'Equation for perpendicular bisector of  CA: ({equation_coeff3[0]:.2f})x + ({equation_coeff3[1]:.2f})y + ({const3:.2f}) = 0')
#
    print("\nQ1.4.2")
    def line_gen(A,B):
        len =10
        dim = A.shape[0]
        x_AB = np.zeros((dim,len))
        lam_1 = np.linspace(0,1,len)
        for i in range(len):
            temp1 = A + lam_1[i]*(B-A)
            x_AB[:,i]= temp1.T
        return x_AB
    F = (A+B)/2
    E = (A+C)/2
    O = line_intersect(B-A,F,C-A,E)
    print("Intersection of AB and AC is: ",O)
#
    print("\nQ1.4.4")
    print("OA = ", np.linalg.norm(O-A))
    print("OB = ", np.linalg.norm(O-B))
    print("OC = ", np.linalg.norm(O-C))
    print("Therefore, OA = OB = OC")
#
    print("\nQ1.4.5")
    print("Figure saved!")
#
    print("\nQ1.4.6")
    dot_pt_O = (B - O) @ ((C - O).T)
    norm_pt_O = np.linalg.norm(B - O) * np.linalg.norm(C - O)
    cos_theta_O = dot_pt_O / norm_pt_O
    angle_BOC = round(360-np.degrees(np.arccos(cos_theta_O)),5)  #Round is used to round of number till 5 decimal places
    print("angle BOC = ", (360-angle_BOC))
    dot_pt_A = (B - A) @ ((C - A).T)
    norm_pt_A = np.linalg.norm(B - A) * np.linalg.norm(C - A)
    cos_theta_A = dot_pt_A / norm_pt_A
    angle_BAC = round(np.degrees(np.arccos(cos_theta_A)),5)  #Round is used to round of number till 5 decimal places
    print("angle BAC = " + str(angle_BAC))
    print("Therefore, angle BOC = 2* angle BAC")

def fig(A,B,C):
    F = (A+B)/2
    E = (A+C)/2
    O = line_intersect(B-A,F,C-A,E)
    #Generating all lines
    x_AB = line_gen(A,B)
    x_BC = line_gen(B,C)
    x_CA = line_gen(C,A)
    x_OE = line_gen(O,E)
    x_OF = line_gen(O,F)
    x_OA = line_gen(O,A)
    #Plotting all lines
    plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
    plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
    plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
    plt.plot(x_OE[0,:],x_OE[1,:],label='$OE$')
    plt.plot(x_OF[0,:],x_OF[1,:],label='$OF$')
    plt.plot(x_OA[0,:],x_OA[1,:],label='$OA$')
    #Generating and plotting circumcircle
    radius = np.linalg.norm(O-A)
    x_ccirc= circ_gen(O,radius)
    plt.plot(x_ccirc[0,:],x_ccirc[1,:],label='$circumcircle$')
    #Labeling the coordinates
    _A = A.reshape(-1,1)
    _B = B.reshape(-1,1)
    _C = C.reshape(-1,1)
    _F = F.reshape(-1,1)
    _E = E.reshape(-1,1)
    _O = O.reshape(-1,1)
    tri_coords = np.block([[_A, _B, _C, _E, _F, _O]])
    plt.scatter(tri_coords[0, :], tri_coords[1, :])
    vert_labels = ['A', 'B', 'C', 'E', 'F', 'O']
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
    plt.savefig('/home/yash/Desktop/geoPython/figs/4.png')
    plt.close()
