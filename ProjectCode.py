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

class FEA:
    
    def __init__(self,mesh,ElementType):
        
        self.nelx = int(mesh['L']/mesh['h'])
        self.nely = int(mesh['H']/mesh['h'])
        self.ElementType = ElementType
        
        mesh['nel'] = self.nelx*self.nely
        
        if ElementType == 'P1' or ElementType == 'P2':
        
            mesh['nel'] = int(mesh['nel']*2)
            
        self.p,self.dofMat,self.mesh = MeshGenerate2D(mesh,ElementType) 
          
    def SolveFEA(self,example):
        
        def ElemStiffnessMatrix1(h,ElemDof):
            
            if ElemDof == 4:                     
                
                aK = np.array([[ h**2/9 + 2/3, h**2/18 - 1/6, h**2/36 - 1/3, h**2/18 - 1/6],\
                               [h**2/18 - 1/6,  h**2/9 + 2/3, h**2/18 - 1/6, h**2/36 - 1/3],\
                               [h**2/36 - 1/3, h**2/18 - 1/6,  h**2/9 + 2/3, h**2/18 - 1/6],\
                               [h**2/18 - 1/6, h**2/36 - 1/3, h**2/18 - 1/6,  h**2/9 + 2/3]])
            elif ElemDof == 3:
              
                aK = np.array([[   h**2/6 + 2, - h**2/12 - 1, - h**2/12 - 1],\
                               [- h**2/12 - 1,    h**2/3 + 1,        h**2/4],\
                               [- h**2/12 - 1,        h**2/4,    h**2/3 + 1]])
                    
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
                
            elif ElemDof == 6:
                
                aK = np.array([[(13*h**2)/30 + 22/3,  (29*h**2)/180 + 7/3,  (29*h**2)/180 + 7/3,  - (3*h**2)/5 - 28/3,    (7*h**2)/9 + 20/3,  - (3*h**2)/5 - 28/3],\
                               [(29*h**2)/180 + 7/3,    (2*h**2)/15 + 7/3,              h**2/36, - (4*h**2)/15 - 14/3,           h**2/3 + 2,     - (2*h**2)/9 - 2],\
                               [(29*h**2)/180 + 7/3,              h**2/36,    (2*h**2)/15 + 7/3,     - (2*h**2)/9 - 2,           h**2/3 + 2, - (4*h**2)/15 - 14/3],\
                               [- (3*h**2)/5 - 28/3, - (4*h**2)/15 - 14/3,     - (2*h**2)/9 - 2,    (44*h**2)/45 + 16, - (10*h**2)/9 - 32/3,    (8*h**2)/9 + 32/3],\
                               [  (7*h**2)/9 + 20/3,           h**2/3 + 2,           h**2/3 + 2, - (10*h**2)/9 - 32/3,   (16*h**2)/9 + 32/3, - (10*h**2)/9 - 32/3],\
                               [- (3*h**2)/5 - 28/3,     - (2*h**2)/9 - 2, - (4*h**2)/15 - 14/3,    (8*h**2)/9 + 32/3, - (10*h**2)/9 - 32/3,    (44*h**2)/45 + 16]])
                
            return aK
        
        
        def ElemStiffnessMatrix2(h,ElemDof):
            
            if ElemDof == 4:                     
                
                aK = np.array([[ 2/3, -1/6, -1/3, -1/6],\
                               [-1/6,  2/3, -1/6, -1/3],\
                               [-1/3, -1/6,  2/3, -1/6],\
                               [-1/6, -1/3, -1/6,  2/3]])
            elif ElemDof == 3:
              
                aK = np.array([[ 2, -1, -1],\
                               [-1,  1,  0],\
                               [-1,  0,  1]])
                    
            elif ElemDof == 9:
                
                aK = np.array([[ 28/45,  -1/30,  -1/45,  -1/30,   -1/5,    1/9,    1/9,   -1/5, -16/45],\
                               [ -1/30,  28/45,  -1/30,  -1/45,   -1/5,   -1/5,    1/9,    1/9, -16/45],\
                               [ -1/45,  -1/30,  28/45,  -1/30,    1/9,   -1/5,   -1/5,    1/9, -16/45],\
                               [ -1/30,  -1/45,  -1/30,  28/45,    1/9,    1/9,   -1/5,   -1/5, -16/45],\
                               [  -1/5,   -1/5,    1/9,    1/9,  88/45, -16/45,      0, -16/45, -16/15],\
                               [   1/9,   -1/5,   -1/5,    1/9, -16/45,  88/45, -16/45,      0, -16/15],\
                               [   1/9,    1/9,   -1/5,   -1/5,      0, -16/45,  88/45, -16/45, -16/15],\
                               [  -1/5,    1/9,    1/9,   -1/5, -16/45,      0, -16/45,  88/45, -16/15],\
                               [-16/45, -16/45, -16/45, -16/45, -16/15, -16/15, -16/15, -16/15, 256/45]],dtype=np.float64)
                
            elif ElemDof == 6:
                
                aK = np.array([[ 22/3,   7/3,   7/3, -28/3,  20/3, -28/3],\
                               [  7/3,   7/3,     0, -14/3,     2,    -2],\
                               [  7/3,     0,   7/3,    -2,     2, -14/3],\
                               [-28/3, -14/3,    -2,    16, -32/3,  32/3],\
                               [ 20/3,     2,     2, -32/3,  32/3, -32/3],\
                               [-28/3,    -2, -14/3,  32/3, -32/3,    16]])
                
            return aK

        def StiffnessMatrix(h,p,s):
            
            ndof =  np.size(p,0)
            A = np.zeros((ndof,ndof))
            
            ElemDof = self.mesh['dofPerElem']
            
            if example == 1:
                aK = ElemStiffnessMatrix1(h,ElemDof)
            else:
                aK = ElemStiffnessMatrix2(h,ElemDof)

            
            nK = np.size(s,0)
            for k in range(nK):       
                for i1 in range(ElemDof):
                    i = s[k,i1]
                    for j1 in range(ElemDof):
                        j = s[k,j1]
                        A[i,j] = A[i,j] + aK[i1,j1]
                     
            return A

        def ForceFun1(x1,y1,h,ElemDof):
            
            if ElemDof == 4:
                F = (1/(2*pi**2))*np.array([[-(cos(pi*(h+x1))-cos(pi*(h+y1))-cos(pi*(x1))+cos(pi*(y1))+h*pi*sin(pi*(x1))-h*pi*sin(pi*(y1)))],\
                              [(cos(pi*(h+x1))+cos(pi*(h+y1))-cos(pi*(x1))-cos(pi*(y1))+h*pi*sin(pi*(x1+h))+h*pi*sin(pi*(y1)))],\
                              [(cos(pi*(h+x1))-cos(pi*(h+y1))-cos(pi*(x1))+cos(pi*(y1))+h*pi*sin(pi*(x1+h))-h*pi*sin(pi*(y1+h)))],\
                              [-(cos(pi*(h+x1))+cos(pi*(h+y1))-cos(pi*(x1))-cos(pi*(y1))+h*pi*sin(pi*(x1))+h*pi*sin(pi*(y1+h)))]])
            
            elif ElemDof == 3:
                
                F = np.array([-(cos(pi*(h + x1)) - cos(pi*(h + y1)) - cos(pi*x1) + cos(pi*y1) + (pi*h*sin(pi*(h + x1)))/2 - (pi*h*sin(pi*(h + y1)))/2 + (h*pi*sin(pi*x1))/2 - (h*pi*sin(pi*y1))/2)/pi**2,\
                                                             (cos(pi*(h + x1)) - cos(pi*x1))/pi**2 - (h*(sin(pi*(h + y1)) - sin(pi*y1)))/(2*pi) + (h*sin(pi*(h + x1)))/pi,\
                                                           cos(pi*y1)/pi**2 - (cos(pi*(h + y1)) + pi*h*sin(pi*(h + y1)))/pi**2 + (h*(sin(pi*(h + x1)) - sin(pi*x1)))/(2*pi)])
            
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
                    
                
            elif ElemDof == 6:
                
                F = np.array([-(24*sin(pi*(h + x1)) - 24*sin(pi*(h + y1)) - 24*sin(pi*x1) + 24*sin(pi*y1) - 7*h**2*pi**2*sin(pi*(h + x1)) + 7*h**2*pi**2*sin(pi*(h + y1)) - 18*pi*h*cos(pi*(h + x1)) + 18*pi*h*cos(pi*(h + y1)) + h**2*pi**2*sin(pi*x1) - h**2*pi**2*sin(pi*y1) - 6*h*pi*cos(pi*x1) + 6*h*pi*cos(pi*y1))/(6*h*pi**3),\
                                                                                                                                      (3*cos(pi*(h + x1)) + cos(pi*x1))/pi**2 - (4*sin(pi*(h + x1)) - 4*sin(pi*x1))/(h*pi**3) + (h*(6*sin(pi*(h + x1)) - sin(pi*(h + y1)) + sin(pi*y1)))/(6*pi),\
                                                                                                                                      (4*sin(pi*(h + y1)) - 4*sin(pi*y1))/(h*pi**3) - (3*cos(pi*(h + y1)) + cos(pi*y1))/pi**2 - (h*(6*sin(pi*(h + y1)) - sin(pi*(h + x1)) + sin(pi*x1)))/(6*pi),\
                                                           -(24*sin(pi*x1) - 24*sin(pi*(h + x1)) + 6*h**2*pi**2*sin(pi*(h + x1)) - 4*h**2*pi**2*sin(pi*(h + y1)) + 18*pi*h*cos(pi*(h + x1)) - 6*pi*h*cos(pi*(h + y1)) - 2*h**2*pi**2*sin(pi*y1) + 6*h*pi*cos(pi*x1) + 6*h*pi*cos(pi*y1))/(3*h*pi**3),\
                                                                                                                                                             (2*cos(pi*(h + x1)) - 2*cos(pi*(h + y1)) - 2*cos(pi*x1) + 2*cos(pi*y1) + 2*pi*h*sin(pi*(h + x1)) - 2*pi*h*sin(pi*(h + y1)))/pi**2,\
                                                            (24*sin(pi*y1) - 24*sin(pi*(h + y1)) - 4*h**2*pi**2*sin(pi*(h + x1)) + 6*h**2*pi**2*sin(pi*(h + y1)) - 6*pi*h*cos(pi*(h + x1)) + 18*pi*h*cos(pi*(h + y1)) - 2*h**2*pi**2*sin(pi*x1) + 6*h*pi*cos(pi*x1) + 6*h*pi*cos(pi*y1))/(3*h*pi**3)])
                
            return F 
        
        def ForceFun2(x1,y1,h,ElemDof):
            
            if ElemDof == 4:

                F = np.array([((sin(pi*x1) - sin(pi*(h + x1)) + h*pi*cos(pi*x1))*(sin(pi*y1) - sin(pi*(h + y1)) + h*pi*cos(pi*y1)))/(h**2*pi**4),\
                              -((sin(pi*x1) - sin(pi*(h + x1)) + pi*h*cos(pi*(h + x1)))*(sin(pi*y1) - sin(pi*(h + y1)) + h*pi*cos(pi*y1)))/(h**2*pi**4),\
                              ((sin(pi*x1) - sin(pi*(h + x1)) + pi*h*cos(pi*(h + x1)))*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h**2*pi**4),\
                              -((sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1)))*(sin(pi*x1) - sin(pi*(h + x1)) + h*pi*cos(pi*x1)))/(h**2*pi**4)])
            
            elif ElemDof == 3:                
                F = np.array([(sin(pi*(2*h + x1 + y1)) + sin(pi*(x1 + y1)) - 2*sin(pi*(h + x1 + y1)) - (h*pi*cos(pi*(2*h + x1 + y1)))/2 + (pi*h*cos(pi*(x1 + y1)))/2)/(h*pi**3),\
                                              ((cos(pi*(h + y1)) - cos(pi*y1))*(sin(pi*x1) - sin(pi*(h + x1)) + pi*h*cos(pi*(h + x1))))/(h*pi**3),\
                                              ((cos(pi*(h + x1)) - cos(pi*x1))*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h*pi**3)])
            
            elif ElemDof == 9:                
                F = np.array([ (16*cos(pi*(2*h + x1 + y1)) - 16*cos(pi*(h - x1 + y1)) - 16*cos(pi*(h + x1 - y1)) + 16*cos(pi*(x1 + y1)) + 32*cos(pi*(x1 - y1)) - 32*cos(pi*(h + x1 + y1)) + 7*h**2*pi**2*cos(pi*(h + x1 - y1)) + 7*h**2*pi**2*cos(pi*(h - x1 + y1)) - h**2*pi**2*cos(pi*(2*h + x1 + y1)) + h**3*pi**3*sin(pi*(h + x1 - y1)) + h**3*pi**3*sin(pi*(h - x1 + y1)) - 17*h**2*pi**2*cos(pi*(x1 + y1)) + h**4*pi**4*cos(pi*(x1 + y1)) + 6*h**3*pi**3*sin(pi*(x1 + y1)) - 16*pi*h*sin(pi*(h + x1 - y1)) - 16*pi*h*sin(pi*(h - x1 + y1)) + 8*h*pi*sin(pi*(2*h + x1 + y1)) - 24*pi*h*sin(pi*(x1 + y1)) + 2*h**2*pi**2*cos(pi*(x1 - y1)) + h**4*pi**4*cos(pi*(x1 - y1)) + 2*h**2*pi**2*cos(pi*(h + x1 + y1)) + 2*h**3*pi**3*sin(pi*(h + x1 + y1)) + 16*pi*h*sin(pi*(h + x1 + y1)))/(2*h**4*pi**6),\
                              -(16*cos(pi*(h + x1 - y1)) + 16*cos(pi*(h - x1 + y1)) - 16*cos(pi*(2*h + x1 + y1)) - 16*cos(pi*(x1 + y1)) - 32*cos(pi*(x1 - y1)) + 32*cos(pi*(h + x1 + y1)) - 17*h**2*pi**2*cos(pi*(h + x1 - y1)) - h**2*pi**2*cos(pi*(h - x1 + y1)) + 7*h**2*pi**2*cos(pi*(2*h + x1 + y1)) + h**4*pi**4*cos(pi*(h + x1 - y1)) - 6*h**3*pi**3*sin(pi*(h + x1 - y1)) + h**3*pi**3*sin(pi*(2*h + x1 + y1)) + 7*h**2*pi**2*cos(pi*(x1 + y1)) - h**3*pi**3*sin(pi*(x1 + y1)) + 24*pi*h*sin(pi*(h + x1 - y1)) + 8*pi*h*sin(pi*(h - x1 + y1)) - 16*h*pi*sin(pi*(2*h + x1 + y1)) + 16*pi*h*sin(pi*(x1 + y1)) + 2*h**2*pi**2*cos(pi*(x1 - y1)) + 2*h**2*pi**2*cos(pi*(h + x1 + y1)) + h**4*pi**4*cos(pi*(h + x1 + y1)) - 2*h**3*pi**3*sin(pi*(x1 - y1)) - 16*pi*h*sin(pi*(x1 - y1)))/(2*h**4*pi**6),\
                                ((4*cos(pi*(h + x1)) - 4*cos(pi*x1) - h**2*pi**2*cos(pi*(h + x1)) + 3*pi*h*sin(pi*(h + x1)) + h*pi*sin(pi*x1))*(4*cos(pi*(h + y1)) - 4*cos(pi*y1) - h**2*pi**2*cos(pi*(h + y1)) + 3*pi*h*sin(pi*(h + y1)) + h*pi*sin(pi*y1)))/(h**4*pi**6),\
                                ((4*cos(pi*(h + y1)) - 4*cos(pi*y1) - h**2*pi**2*cos(pi*(h + y1)) + 3*pi*h*sin(pi*(h + y1)) + h*pi*sin(pi*y1))*(4*cos(pi*(h + x1)) - 4*cos(pi*x1) + h**2*pi**2*cos(pi*x1) + pi*h*sin(pi*(h + x1)) + 3*h*pi*sin(pi*x1)))/(h**4*pi**6),\
                                -(16*cos(pi*(2*h + x1 + y1)) - 16*cos(pi*(h - x1 + y1)) - 16*cos(pi*(h + x1 - y1)) + 16*cos(pi*(x1 + y1)) + 32*cos(pi*(x1 - y1)) - 32*cos(pi*(h + x1 + y1)) + 10*h**2*pi**2*cos(pi*(h + x1 - y1)) + 2*h**2*pi**2*cos(pi*(h - x1 + y1)) - 2*h**2*pi**2*cos(pi*(2*h + x1 + y1)) + 2*h**3*pi**3*sin(pi*(h + x1 - y1)) - 10*h**2*pi**2*cos(pi*(x1 + y1)) + 2*h**3*pi**3*sin(pi*(x1 + y1)) - 20*pi*h*sin(pi*(h + x1 - y1)) - 12*pi*h*sin(pi*(h - x1 + y1)) + 12*h*pi*sin(pi*(2*h + x1 + y1)) - 20*pi*h*sin(pi*(x1 + y1)) + 4*h**2*pi**2*cos(pi*(x1 - y1)) - 4*h**2*pi**2*cos(pi*(h + x1 + y1)) + 2*h**3*pi**3*sin(pi*(x1 - y1)) + 2*h**3*pi**3*sin(pi*(h + x1 + y1)) + 8*pi*h*sin(pi*(x1 - y1)) + 8*pi*h*sin(pi*(h + x1 + y1)))/(h**4*pi**6),\
                                -(4*(2*cos(pi*(h + y1)) - 2*cos(pi*y1) + pi*h*sin(pi*(h + y1)) + h*pi*sin(pi*y1))*(4*cos(pi*(h + x1)) - 4*cos(pi*x1) - h**2*pi**2*cos(pi*(h + x1)) + 3*pi*h*sin(pi*(h + x1)) + h*pi*sin(pi*x1)))/(h**4*pi**6),\
                                -(4*(2*cos(pi*(h + x1)) - 2*cos(pi*x1) + pi*h*sin(pi*(h + x1)) + h*pi*sin(pi*x1))*(4*cos(pi*(h + y1)) - 4*cos(pi*y1) - h**2*pi**2*cos(pi*(h + y1)) + 3*pi*h*sin(pi*(h + y1)) + h*pi*sin(pi*y1)))/(h**4*pi**6),\
                                -(4*(2*cos(pi*(h + y1)) - 2*cos(pi*y1) + pi*h*sin(pi*(h + y1)) + h*pi*sin(pi*y1))*(4*cos(pi*(h + x1)) - 4*cos(pi*x1) + h**2*pi**2*cos(pi*x1) + pi*h*sin(pi*(h + x1)) + 3*h*pi*sin(pi*x1)))/(h**4*pi**6),\
                                -(32*cos(pi*(h + x1 - y1)) + 32*cos(pi*(h - x1 + y1)) - 32*cos(pi*(2*h + x1 + y1)) - 32*cos(pi*(x1 + y1)) - 64*cos(pi*(x1 - y1)) + 64*cos(pi*(h + x1 + y1)) - 8*h**2*pi**2*cos(pi*(h + x1 - y1)) - 8*h**2*pi**2*cos(pi*(h - x1 + y1)) + 8*h**2*pi**2*cos(pi*(2*h + x1 + y1)) + 8*h**2*pi**2*cos(pi*(x1 + y1)) + 32*pi*h*sin(pi*(h + x1 - y1)) + 32*pi*h*sin(pi*(h - x1 + y1)) - 32*h*pi*sin(pi*(2*h + x1 + y1)) + 32*pi*h*sin(pi*(x1 + y1)) - 16*h**2*pi**2*cos(pi*(x1 - y1)) + 16*h**2*pi**2*cos(pi*(h + x1 + y1)))/(h**4*pi**6)])
                
            elif ElemDof == 6:
                F = np.array([ (2*cos(pi*(h + x1 - y1)) + 2*cos(pi*(h - x1 + y1)) - 6*cos(pi*(2*h + x1 + y1)) - 6*cos(pi*(x1 + y1)) - 4*cos(pi*(x1 - y1)) + 12*cos(pi*(h + x1 + y1)) + (3*h**2*pi**2*cos(pi*(2*h + x1 + y1)))/2 + (h**2*pi**2*cos(pi*(x1 + y1)))/2 - 5*h*pi*sin(pi*(2*h + x1 + y1)) + 3*pi*h*sin(pi*(x1 + y1)) + 2*h**2*pi**2*cos(pi*(x1 - y1)) + 2*pi*h*sin(pi*(h + x1 + y1)))/(h**2*pi**4),\
                              -((cos(pi*(h + y1)) - cos(pi*y1))*(4*cos(pi*(h + x1)) - 4*cos(pi*x1) - h**2*pi**2*cos(pi*(h + x1)) + 3*pi*h*sin(pi*(h + x1)) + h*pi*sin(pi*x1)))/(h**2*pi**4),\
                              -((cos(pi*(h + x1)) - cos(pi*x1))*(4*cos(pi*(h + y1)) - 4*cos(pi*y1) - h**2*pi**2*cos(pi*(h + y1)) + 3*pi*h*sin(pi*(h + y1)) + h*pi*sin(pi*y1)))/(h**2*pi**4),\
                              (4*sin(pi*x1)*(cos(pi*(h + y1)) - cos(pi*y1)))/(h*pi**3) - (8*cos(pi*x1)*(cos(pi*(h + y1)) - cos(pi*y1)))/(h**2*pi**4) - (4*cos(pi*(h + x1))*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h*pi**3) + (4*sin(pi*(h + x1))*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h**2*pi**4) - (4*sin(pi*x1)*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h**2*pi**4) + (8*cos(pi*(h + x1))*(cos(pi*(h + y1)) - cos(pi*y1)))/(h**2*pi**4) + (4*sin(pi*(h + x1))*(cos(pi*(h + y1)) - cos(pi*y1)))/(h*pi**3),\
                              (4*(sin(pi*x1) - sin(pi*(h + x1)) + pi*h*cos(pi*(h + x1)))*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h**2*pi**4),\
                              (4*sin(pi*(h + x1))*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h**2*pi**4) - (4*cos(pi*x1)*(2*cos(pi*(h + y1)) - 2*cos(pi*y1) - h**2*pi**2*cos(pi*(h + y1)) + 2*pi*h*sin(pi*(h + y1))))/(h**2*pi**4) - (4*cos(pi*x1)*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h*pi**3) - (4*sin(pi*x1)*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h**2*pi**4) + (4*cos(pi*(h + x1))*(2*cos(pi*(h + y1)) - 2*cos(pi*y1) - h**2*pi**2*cos(pi*(h + y1)) + 2*pi*h*sin(pi*(h + y1))))/(h**2*pi**4)])
                
            return F

        def BoundaryCondition(h,p,s):
                       
            ndof =  np.size(p,0)
            dofs = np.arange(0,ndof)

            
            if example == 1:
                fixedNodes=[]
            else:
                find = np.isclose(p[:,0],0,rtol=1e-6)*1
                find = np.where(find==1)
                fixedNodeLeft = find[0].astype(int)     
                
                find = np.isclose(p[:,1],0,rtol=1e-6)*1
                find = np.where(find==1)
                fixedNodeBottom = find[0].astype(int)    
                
                find = np.isclose(p[:,0],np.max(p[:,0]),rtol=1e-6)*1
                find = np.where(find==1)
                fixedNodeRight = find[0].astype(int)  
                
                find = np.isclose(p[:,1],np.max(p[:,1]),rtol=1e-6)*1
                find = np.where(find==1)
                fixedNodeTop = find[0].astype(int)  
                
                fixedNodes = np.concatenate((fixedNodeRight,fixedNodeLeft,\
                                             fixedNodeBottom,fixedNodeTop)).T.reshape(-1);
            
            
            
            freeNodes = np.setdiff1d(dofs,fixedNodes) 
                        
            u = np.zeros(ndof)
            b = np.zeros((ndof,1))

            
            nK = np.size(s,0)
            ElemDof = self.mesh['dofPerElem']
            for k in range(nK):
                
                x1 = p[s[k,0],0]
                y1 = p[s[k,0],1]
                        
                if example == 1:
                    F = ForceFun1(x1,y1,h,ElemDof)
                else:
                    F = ForceFun2(x1,y1,h,ElemDof)

                for i1 in range(ElemDof):
                    i = s[k,i1]
                    
                    b[i,0] = b[i,0] + F[i1]
        
            return u, b, freeNodes
            
        A = StiffnessMatrix(self.mesh['h'],self.p,self.dofMat) # stiffness matrix
                
        u,b,freeNodes = BoundaryCondition(self.mesh['h'],self.p,self.dofMat) # boundary condtions

        u[freeNodes] = solve(A[freeNodes,:][:,freeNodes],b[freeNodes,0])

        # self.Plot_Solution(u,'Element ' + self.ElementType + ' FEA Solution for Elements ' )
                
        if example == 1:
            uref = (cos(pi*self.p[:,0])- cos(pi*self.p[:,1]))/(1+pi**2)
        else:            
            uref = sin(self.p[:,0]*pi)*sin(self.p[:,1]*pi)/(2*pi**2)
                        
        # self.Plot_Solution(uref,'Exact Solution for Elements ')

        normL2  = norm(uref-u,ord=2)
        normLinf = norm(uref-u,ord=np.inf)

        return u,normL2,normLinf
        
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
    


h = 1/2**2
example = 2
ElementType = 'Q1'

L = 1 
H = 1

mesh = {'h':h,'L':L,'H':H}

Type = ['P1','P2','Q1','Q2']

num = len(Type)
meshNum = 7

L2norm = np.zeros((num,meshNum-1))
Linfnorm = np.zeros((num,meshNum-1))
H = 1/2**np.arange(1,meshNum)
for i in range(num):
    ElementType = Type[i]
    for j in range(meshNum-1):
        mesh['h'] = H[j]
        F = FEA(mesh,ElementType)
        u,normL2,normLinf = F.SolveFEA(example)
        L2norm[i,j]   = normL2
        Linfnorm[i,j] = normLinf
    
    plt.figure(1)
    plt.loglog(H,L2norm[i,:],'-*')
    plt.figure(2)
    plt.loglog(H,Linfnorm[i,:],'-o')

plt.figure(1)
plt.xlabel('Log(Element size=h)',fontsize=16)
plt.ylabel('Log($L_{2}$ norm of Error)', fontsize=16)
plt.title('$L_{2}$ Norm Error vs Element size', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(Type,fontsize=16)

plt.figure(2)
plt.xlabel('Log(Element size=h)',fontsize=16)
plt.ylabel('Log($L_{\infty}$ norm of Error)', fontsize=16)
plt.title('$L_{\infty}$ Norm Error vs Element size', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(Type,fontsize=16)

plt.show()
    
    
