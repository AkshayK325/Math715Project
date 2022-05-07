# importing numpy
import numpy as np
# importing sparse diags
from scipy.sparse import spdiags

# importing sparse solver
from scipy.linalg import lu,svd,solve_triangular,eig,null_space
from numpy.linalg import qr,solve,cholesky,norm,cond

# importing the ploting libray
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve,gmres,spsolve_triangular

import sys
from sys import exit
from time import time

from numpy import pi, sin, cos

from utillsFuncs import MeshGenerate2D


h = 0.25
L = 1 
H = 1

ElementType = 'Q2'

mesh = {'h':h,'L':L,'H':H}


class FEA:
    
    def __init__(self,mesh,ElementType):
        
        self.nelx = int(mesh['L']/mesh['h'])
        self.nely = int(mesh['H']/mesh['h'])
        
        mesh['nel'] = self.nelx*self.nely
        
        if ElementType == 'P1' or ElementType == 'P2':
        
            mesh['nel'] = int(mesh['nel']*2)
            
        self.p,self.dofMat,self.mesh = MeshGenerate2D(mesh,ElementType) 
          
    
    def SolveFEA(self,bc):
        
        def ElemStiffnessMatrix(h,ElemDof):
            
            if ElemDof == 4:
                t1 = h**2 / 9
                t2 = t1/2
                t3 = t2/2
                
                c1 = 2/3
                c2 = 1/6
                c3 = 1/3            
                
                aK = np.array([[t1+c1,t2-c2,t3-c3,t2-c2],\
                               [t2-c2,t1+c1,t2-c2,t3-c3],\
                               [t3-c3,t2-c2,t1+c1,t2-c2],\
                               [t2-c2,t3-c3,t2-c2,t1+c1]])
            elif ElemDof == 3:
                
                c1 = (7*h**2)/6 - 2*h + 3
                c2 = h/2 - (7*h**2)/12 - 1
                c3 = h**2/3 + 1
                c4 = h**2/4
                
                aK = np.array([[c1,c2,c2],\
                               [c2,c3,c4],\
                               [c2,c4,c3]])
                    
            elif ElemDof == 9:
                
                aK = np.array([[      (4*h**2)/225 + 28/45,          - h**2/225 - 1/30, -(- h**8/900 + h**6/45)/h**6,          - h**2/225 - 1/30,               (2*h**2)/225 - 1/5,                   1/9 - h**2/450,                   1/9 - h**2/450,               (2*h**2)/225 - 1/5,                 h**2/225 - 16/45],\
                                [         - h**2/225 - 1/30,       (4*h**2)/225 + 28/45,          - h**2/225 - 1/30, -(- h**8/900 + h**6/45)/h**6,               (2*h**2)/225 - 1/5,               (2*h**2)/225 - 1/5,                   1/9 - h**2/450,                   1/9 - h**2/450,                 h**2/225 - 16/45],\
                                [-(- h**8/900 + h**6/45)/h**6,          - h**2/225 - 1/30,       (4*h**2)/225 + 28/45,          - h**2/225 - 1/30,                   1/9 - h**2/450,               (2*h**2)/225 - 1/5,               (2*h**2)/225 - 1/5,                   1/9 - h**2/450,                 h**2/225 - 16/45],\
                                [         - h**2/225 - 1/30, -(- h**8/900 + h**6/45)/h**6,          - h**2/225 - 1/30,       (4*h**2)/225 + 28/45,                   1/9 - h**2/450,                   1/9 - h**2/450,               (2*h**2)/225 - 1/5,               (2*h**2)/225 - 1/5,                 h**2/225 - 16/45],\
                                [        (2*h**2)/225 - 1/5,         (2*h**2)/225 - 1/5,             1/9 - h**2/450,             1/9 - h**2/450,            (16*h**2)/225 + 88/45,                 h**2/225 - 16/45,                    -(4*h**2)/225,                 h**2/225 - 16/45, -(8*(- h**8 + 30*h**6))/(225*h**6)],\
                                [            1/9 - h**2/450,         (2*h**2)/225 - 1/5,         (2*h**2)/225 - 1/5,             1/9 - h**2/450,                 h**2/225 - 16/45,            (16*h**2)/225 + 88/45,                 h**2/225 - 16/45,                    -(4*h**2)/225, -(8*(- h**8 + 30*h**6))/(225*h**6)],\
                                [            1/9 - h**2/450,             1/9 - h**2/450,         (2*h**2)/225 - 1/5,         (2*h**2)/225 - 1/5,                    -(4*h**2)/225,                 h**2/225 - 16/45,            (16*h**2)/225 + 88/45,                 h**2/225 - 16/45, -(8*(- h**8 + 30*h**6))/(225*h**6)],\
                                [        (2*h**2)/225 - 1/5,             1/9 - h**2/450,             1/9 - h**2/450,         (2*h**2)/225 - 1/5,                 h**2/225 - 16/45,                    -(4*h**2)/225,                 h**2/225 - 16/45,            (16*h**2)/225 + 88/45, -(8*(- h**8 + 30*h**6))/(225*h**6)],\
                                [          h**2/225 - 16/45,           h**2/225 - 16/45,           h**2/225 - 16/45,           h**2/225 - 16/45, -(8*(- h**8 + 30*h**6))/(225*h**6), -(8*(- h**8 + 30*h**6))/(225*h**6), -(8*(- h**8 + 30*h**6))/(225*h**6), -(8*(- h**8 + 30*h**6))/(225*h**6),           (64*h**2)/225 + 256/45]],dtype=np.float64)
                                                
                
            return aK

        def StiffnessMatrix(h,p,s):
            
            ndof =  np.size(p,0)
            A = np.zeros((ndof,ndof))
            
            ElemDof = self.mesh['dofPerElem']

            aK = ElemStiffnessMatrix(h,ElemDof)

            nK = np.size(s,0)
            for k in range(nK):       
                for i1 in range(np.size(aK,0)):
                    i = s[k,i1]
                    for j1 in range(np.size(aK,0)):
                        j = s[k,j1]
                        A[i,j] = A[i,j] + aK[i1,j1]
                     
            return A

        def ForceFun(x1,y1,h,ElemDof):
            
            if ElemDof == 4:
                F = (1/(2*pi**2))*np.array([[-(cos(pi*(h+x1))-cos(pi*(h+y1))-cos(pi*(x1))+cos(pi*(y1))+h*pi*sin(pi*(x1))-h*pi*sin(pi*(y1)))],\
                              [(cos(pi*(h+x1))+cos(pi*(h+y1))-cos(pi*(x1))-cos(pi*(y1))+h*pi*sin(pi*(x1+h))+h*pi*sin(pi*(y1)))],\
                              [(cos(pi*(h+x1))-cos(pi*(h+y1))-cos(pi*(x1))+cos(pi*(y1))+h*pi*sin(pi*(x1+h))-h*pi*sin(pi*(y1+h)))],\
                              [-(cos(pi*(h+x1))+cos(pi*(h+y1))-cos(pi*(x1))-cos(pi*(y1))+h*pi*sin(pi*(x1))+h*pi*sin(pi*(y1+h)))]])
            
            elif ElemDof == 3:
                
                F = (-1/(2*pi**2))*np.array([[2*cos(pi*(h + x1)) - 2*cos(pi*(h + y1)) - 2*cos(pi*x1) + 2*cos(pi*y1) - 2*pi*sin(pi*(h + x1)) + 2*pi*sin(pi*(h + y1)) + 2*pi*sin(pi*x1) - 2*pi*sin(pi*y1) + 3*pi*h*sin(pi*(h + x1)) - 3*pi*h*sin(pi*(h + y1)) - h*pi*sin(pi*x1) + h*pi*sin(pi*y1)],\
                              [2*pi*h*sin(pi*(h+x1))+2*cos(pi*(h+x1))+pi*h*sin(pi*y1)-pi*h*sin(pi*(h+y1))-2*cos(pi*x1)],\
                                  [-pi* h*sin(pi* x1) + pi* h*sin(pi* h*+ pi* x1) - 2*pi* h*sin(pi* h + pi* y1) - 2*cos(pi*(h + y1)) + 2*cos(pi*y1)]])
            
            elif ElemDof == 9:
                
                F = np.array([-(4*sin(pi*(h + x1)) - 4*sin(pi*(h + y1)) - 4*sin(pi*x1) + 4*sin(pi*y1) - pi*h*cos(pi*(h + x1)) + pi*h*cos(pi*(h + y1)) + h**2*pi**2*sin(pi*x1) - h**2*pi**2*sin(pi*y1) - 3*h*pi*cos(pi*x1) + 3*h*pi*cos(pi*y1))/(6*h*pi**3),\
                              (4*sin(pi*(h + y1)) - 4*sin(pi*(h + x1)) + 4*sin(pi*x1) - 4*sin(pi*y1) + h**2*pi**2*sin(pi*(h + x1)) + 3*pi*h*cos(pi*(h + x1)) - pi*h*cos(pi*(h + y1)) + h**2*pi**2*sin(pi*y1) + h*pi*cos(pi*x1) - 3*h*pi*cos(pi*y1))/(6*h*pi**3),\
                              -(4*sin(pi*(h + x1)) - 4*sin(pi*(h + y1)) - 4*sin(pi*x1) + 4*sin(pi*y1) - h**2*pi**2*sin(pi*(h + x1)) + h**2*pi**2*sin(pi*(h + y1)) - 3*pi*h*cos(pi*(h + x1)) + 3*pi*h*cos(pi*(h + y1)) - h*pi*cos(pi*x1) + h*pi*cos(pi*y1))/(6*h*pi**3),\
                              -(4*sin(pi*(h + x1)) - 4*sin(pi*(h + y1)) - 4*sin(pi*x1) + 4*sin(pi*y1) + h**2*pi**2*sin(pi*(h + y1)) - pi*h*cos(pi*(h + x1)) + 3*pi*h*cos(pi*(h + y1)) + h**2*pi**2*sin(pi*x1) - 3*h*pi*cos(pi*x1) + h*pi*cos(pi*y1))/(6*h*pi**3),\
                              -(4*sin(pi*x1) - 8*sin(pi*(h + y1)) - 4*sin(pi*(h + x1)) + 8*sin(pi*y1) + 2*pi*h*cos(pi*(h + x1)) + 2*pi*h*cos(pi*(h + y1)) - 2*h**2*pi**2*sin(pi*y1) + 2*h*pi*cos(pi*x1) + 6*h*pi*cos(pi*y1))/(3*h*pi**3),\
                              (8*sin(pi*x1) - 4*sin(pi*(h + y1)) - 8*sin(pi*(h + x1)) + 4*sin(pi*y1) + 2*h**2*pi**2*sin(pi*(h + x1)) + 6*pi*h*cos(pi*(h + x1)) + 2*pi*h*cos(pi*(h + y1)) + 2*h*pi*cos(pi*x1) + 2*h*pi*cos(pi*y1))/(3*h*pi**3),\
                              -(4*sin(pi*x1) - 8*sin(pi*(h + y1)) - 4*sin(pi*(h + x1)) + 8*sin(pi*y1) + 2*h**2*pi**2*sin(pi*(h + y1)) + 2*pi*h*cos(pi*(h + x1)) + 6*pi*h*cos(pi*(h + y1)) + 2*h*pi*cos(pi*x1) + 2*h*pi*cos(pi*y1))/(3*h*pi**3),\
                              (8*sin(pi*x1) - 4*sin(pi*(h + y1)) - 8*sin(pi*(h + x1)) + 4*sin(pi*y1) + 2*pi*h*cos(pi*(h + x1)) + 2*pi*h*cos(pi*(h + y1)) - 2*h**2*pi**2*sin(pi*x1) + 6*h*pi*cos(pi*x1) + 2*h*pi*cos(pi*y1))/(3*h*pi**3),\
                              (16*sin(pi*(h + x1)) - 16*sin(pi*(h + y1)) - 16*sin(pi*x1) + 16*sin(pi*y1) - 8*pi*h*cos(pi*(h + x1)) + 8*pi*h*cos(pi*(h + y1)) - 8*h*pi*cos(pi*x1) + 8*h*pi*cos(pi*y1))/(3*h*pi**3)])
                    
                
                
                
            return F 

        def BoundaryCondition(h,p,s):
                       
            ndof =  np.size(p,0)
            dof = np.arange(0,ndof)

            b = np.zeros((ndof,1))

            freeNodes = dof.copy()

            u = np.zeros(ndof)
            
            nK = np.size(s,0)
            ElemDof = self.mesh['dofPerElem']
            for k in range(nK):
                
                x1 = p[s[k,0],0]
                y1 = p[s[k,0],1]
                        
                F = ForceFun(x1,y1,h,ElemDof)
                
                for i1 in range(ElemDof):
                    i = s[k,i1]
                    
                    b[i,0] = b[i,0] + F[i1]
        
            return u, b, freeNodes
        
    
        A = StiffnessMatrix(self.mesh['h'],self.p,self.dofMat) # stiffness matrix
        
        u,b,freeNodes = BoundaryCondition(self.mesh['h'],self.p,self.dofMat) # boundary condtions

        u[freeNodes] = solve(A[freeNodes,:][:,freeNodes],b[freeNodes,0])

        self.Plot_Solution(u,'FEA Solution for Elements ')
        
        return u
        
    def Plot_Solution(self,u,TitleStr):
        
        Ny = int(np.sqrt(np.size(self.p,0)))
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.plot_surface(self.p[:,0].reshape((-1,Ny)),self.p[:,1].reshape((-1,Ny)),u.reshape((-1,Ny)),cmap=plt.cm.viridis)
    
        plt.xlabel("x axis")
        plt.ylabel("y axis",rotation = "vertical",labelpad = 20)
        plt.title(TitleStr + str(np.size(self.dofMat,0)))
    
        plt.ylim((np.min(self.p[:]),np.max(self.p[:])))
        plt.xlim((np.min(self.p[:]),np.max(self.p[:])))
    
        plt.show()
    
        
F = FEA(mesh,ElementType)

u = F.SolveFEA(bc=0)