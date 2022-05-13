# importing numpy
import numpy as np
# importing sparse diags
from scipy.sparse import spdiags

# importing sparse solver
from scipy.linalg import lu,svd,solve_triangular,eig,null_space
from numpy.linalg import qr,solve,cholesky,norm,cond

# importing the ploting libray
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix, diags,coo_matrix
from scipy.sparse.linalg import spsolve,gmres,spsolve_triangular,cg

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
              
        
        self.iK = tuple(np.kron(self.dofMat,np.ones((self.mesh['dofPerElem'],1))).flatten().astype(int))
         
        self.jK = tuple(np.kron(self.dofMat,np.ones((1,self.mesh['dofPerElem']))).flatten().astype(int))
            
    def SolveFEA(self,example,Plot_it):
        
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
                               [(29*h**2)/180 + 7/3,              (h**2)/36,    (2*h**2)/15 + 7/3,     - (2*h**2)/9 - 2,           h**2/3 + 2, - (4*h**2)/15 - 14/3],\
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

        def StiffnessMatrixAssembly(h,p,s):
            
            ndof =  np.size(p,0)
            # A = np.zeros((ndof,ndof))
            
            # A = csc_matrix((ndof,ndof),dtype=np.float64)
            
            ElemDof = self.mesh['dofPerElem']
          
            nK = np.size(s,0)
           
            if example == 1:
                aK = ElemStiffnessMatrix1(h,ElemDof)
            else:
                aK = ElemStiffnessMatrix2(h,ElemDof)
                
            Ael = np.tile(aK,(nK,1,1))
                                               
            saK = ((Ael.flatten())).flatten(order='F')
    
            A = coo_matrix((saK,(self.iK,self.jK)),shape=(ndof,ndof)).tocsc()
                        
            return A

        def ForceFun1(x1,y1,h,ElemDof,num):
            
            if ElemDof == 4:
                F = (1/(2*pi**2))*np.array([[-(cos(pi*(h+x1))-cos(pi*(h+y1))-cos(pi*(x1))+cos(pi*(y1))+h*pi*sin(pi*(x1))-h*pi*sin(pi*(y1)))],\
                              [(cos(pi*(h+x1))+cos(pi*(h+y1))-cos(pi*(x1))-cos(pi*(y1))+h*pi*sin(pi*(x1+h))+h*pi*sin(pi*(y1)))],\
                              [(cos(pi*(h+x1))-cos(pi*(h+y1))-cos(pi*(x1))+cos(pi*(y1))+h*pi*sin(pi*(x1+h))-h*pi*sin(pi*(y1+h)))],\
                              [-(cos(pi*(h+x1))+cos(pi*(h+y1))-cos(pi*(x1))-cos(pi*(y1))+h*pi*sin(pi*(x1))+h*pi*sin(pi*(y1+h)))]])
            
            elif ElemDof == 3:
                
                if num%2==0:
                    F = np.array([-(cos(pi*(h + x1)) - cos(pi*(h + y1)) - cos(pi*x1) + cos(pi*y1) + (pi*h*sin(pi*(h + x1)))/2 - (pi*h*sin(pi*(h + y1)))/2 + (h*pi*sin(pi*x1))/2 - (h*pi*sin(pi*y1))/2)/pi**2,\
                                                             (cos(pi*(h + x1)) - cos(pi*x1))/pi**2 - (h*(sin(pi*(h + y1)) - sin(pi*y1)))/(2*pi) + (h*sin(pi*(h + x1)))/pi,\
                                                           cos(pi*y1)/pi**2 - (cos(pi*(h + y1)) + pi*h*sin(pi*(h + y1)))/pi**2 + (h*(sin(pi*(h + x1)) - sin(pi*x1)))/(2*pi)])
                else:
                    
                    F = np.array([(cos(pi*x1) - cos(pi*y1) - cos(pi*(h - x1)) + cos(pi*(h - y1)) + (h*pi*sin(pi*x1))/2 - (h*pi*sin(pi*y1))/2 - (pi*h*sin(pi*(h - x1)))/2 + (pi*h*sin(pi*(h - y1)))/2)/pi**2,\
                                                             (h*sin(pi*(h - x1)))/pi - (h*(sin(pi*y1) + sin(pi*(h - y1))))/(2*pi) - (cos(pi*x1) - cos(pi*(h - x1)))/pi**2,\
                                                          cos(pi*y1)/pi**2 - (cos(pi*(h - y1)) + pi*h*sin(pi*(h - y1)))/pi**2 + (h*(sin(pi*x1) + sin(pi*(h - x1))))/(2*pi)])
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
                if num%2 == 0:
                    F = np.array([-(4*sin(pi*(h + x1)) - 4*sin(pi*(h + y1)) - 4*sin(pi*x1) + 4*sin(pi*y1) - (7*h**2*pi**2*sin(pi*(h + x1)))/6 + (7*h**2*pi**2*sin(pi*(h + y1)))/6 - 3*pi*h*cos(pi*(h + x1)) + 3*pi*h*cos(pi*(h + y1)) + (h**2*pi**2*sin(pi*x1))/6 - (h**2*pi**2*sin(pi*y1))/6 - h*pi*cos(pi*x1) + h*pi*cos(pi*y1))/(h*pi**3),\
                              (3*cos(pi*(h + x1)) + cos(pi*x1))/pi**2 - (4*sin(pi*(h + x1)) - 4*sin(pi*x1))/(h*pi**3) + (h*(6*sin(pi*(h + x1)) - sin(pi*(h + y1)) + sin(pi*y1)))/(6*pi),\
                              -(4*sin(pi*y1) - 4*sin(pi*(h + y1)) - (h**2*pi**2*sin(pi*(h + x1)))/6 + h**2*pi**2*sin(pi*(h + y1)) + 3*pi*h*cos(pi*(h + y1)) + (h**2*pi**2*sin(pi*x1))/6 + h*pi*cos(pi*y1))/(h*pi**3),\
                              -(2*(12*sin(pi*x1) - 12*sin(pi*(h + x1)) + 3*h**2*pi**2*sin(pi*(h + x1)) - 2*h**2*pi**2*sin(pi*(h + y1)) + 9*pi*h*cos(pi*(h + x1)) - 3*pi*h*cos(pi*(h + y1)) - h**2*pi**2*sin(pi*y1) + 3*h*pi*cos(pi*x1) + 3*h*pi*cos(pi*y1)))/(3*h*pi**3),\
                              (2*(cos(pi*(h + x1)) - cos(pi*(h + y1)) - cos(pi*x1) + cos(pi*y1) + pi*h*sin(pi*(h + x1)) - pi*h*sin(pi*(h + y1))))/pi**2,\
                              (6*cos(pi*(h + y1)) - 2*cos(pi*(h + x1)) + 2*cos(pi*x1) + 2*cos(pi*y1))/pi**2 - (8*sin(pi*(h + y1)) - 8*sin(pi*y1))/(h*pi**3) - (2*h*(2*sin(pi*(h + x1)) - 3*sin(pi*(h + y1)) + sin(pi*x1)))/(3*pi)])
                else:
                    F = np.array([-(4*sin(pi*x1) - 4*sin(pi*y1) + 4*sin(pi*(h - x1)) - 4*sin(pi*(h - y1)) - (h**2*pi**2*sin(pi*x1))/6 + (h**2*pi**2*sin(pi*y1))/6 - (7*h**2*pi**2*sin(pi*(h - x1)))/6 + (7*h**2*pi**2*sin(pi*(h - y1)))/6 - h*pi*cos(pi*x1) + h*pi*cos(pi*y1) - 3*pi*h*cos(pi*(h - x1)) + 3*pi*h*cos(pi*(h - y1)))/(h*pi**3),\
                                  (cos(pi*x1) + 3*cos(pi*(h - x1)))/pi**2 - (4*sin(pi*x1) + 4*sin(pi*(h - x1)))/(h*pi**3) - (h*(sin(pi*y1) - 6*sin(pi*(h - x1)) + sin(pi*(h - y1))))/(6*pi),\
                                  (4*sin(pi*y1) + 4*sin(pi*(h - y1)) + (h**2*pi**2*sin(pi*x1))/6 + (h**2*pi**2*sin(pi*(h - x1)))/6 - h**2*pi**2*sin(pi*(h - y1)) - h*pi*cos(pi*y1) - 3*pi*h*cos(pi*(h - y1)))/(h*pi**3),\
                                  -(2*(h**2*pi**2*sin(pi*y1) - 12*sin(pi*(h - x1)) - 12*sin(pi*x1) + 3*h**2*pi**2*sin(pi*(h - x1)) - 2*h**2*pi**2*sin(pi*(h - y1)) + 3*h*pi*cos(pi*x1) + 3*h*pi*cos(pi*y1) + 9*pi*h*cos(pi*(h - x1)) - 3*pi*h*cos(pi*(h - y1))))/(3*h*pi**3),\
                                  -(2*(cos(pi*x1) - cos(pi*y1) - cos(pi*(h - x1)) + cos(pi*(h - y1)) - pi*h*sin(pi*(h - x1)) + pi*h*sin(pi*(h - y1))))/pi**2,\
                                  (2*cos(pi*x1) + 2*cos(pi*y1) - 2*cos(pi*(h - x1)) + 6*cos(pi*(h - y1)))/pi**2 - (8*sin(pi*y1) + 8*sin(pi*(h - y1)))/(h*pi**3) + (2*h*(sin(pi*x1) - 2*sin(pi*(h - x1)) + 3*sin(pi*(h - y1))))/(3*pi)])
                
            return F
        
        def ForceFun2(x1,y1,h,ElemDof,num):
            
            if ElemDof == 4:
                F = np.array([((sin(pi*x1) - sin(pi*(h + x1)) + h*pi*cos(pi*x1))*(sin(pi*y1) - sin(pi*(h + y1)) + h*pi*cos(pi*y1)))/(h**2*pi**4),\
                              -((sin(pi*x1) - sin(pi*(h + x1)) + pi*h*cos(pi*(h + x1)))*(sin(pi*y1) - sin(pi*(h + y1)) + h*pi*cos(pi*y1)))/(h**2*pi**4),\
                              ((sin(pi*x1) - sin(pi*(h + x1)) + pi*h*cos(pi*(h + x1)))*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h**2*pi**4),\
                              -((sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1)))*(sin(pi*x1) - sin(pi*(h + x1)) + h*pi*cos(pi*x1)))/(h**2*pi**4)])
            
            elif ElemDof == 3:  
                if num%2==0:
                    F = np.array([(sin(pi*(2*h + x1 + y1)) + sin(pi*(x1 + y1)) - 2*sin(pi*(h + x1 + y1)) - (h*pi*cos(pi*(2*h + x1 + y1)))/2 + (pi*h*cos(pi*(x1 + y1)))/2)/(h*pi**3),\
                                                  ((cos(pi*(h + y1)) - cos(pi*y1))*(sin(pi*x1) - sin(pi*(h + x1)) + pi*h*cos(pi*(h + x1))))/(h*pi**3),\
                                                  ((cos(pi*(h + x1)) - cos(pi*x1))*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h*pi**3)])
                else:
                    F = np.array([-(sin(pi*(x1 - 2*h + y1)) - 2*sin(pi*(x1 - h + y1)) + sin(pi*(x1 + y1)) + (h*pi*cos(pi*(x1 - 2*h + y1)))/2 - (pi*h*cos(pi*(x1 + y1)))/2)/(h*pi**3),\
                                  ((cos(pi*y1) - cos(pi*(h - y1)))*(sin(pi*x1) + sin(pi*(h - x1)) - pi*h*cos(pi*(h - x1))))/(h*pi**3),\
                                  ((cos(pi*x1) - cos(pi*(h - x1)))*(sin(pi*y1) + sin(pi*(h - y1)) - pi*h*cos(pi*(h - y1))))/(h*pi**3)])
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
                
                if num%2 == 0:
                    F = np.array([(2*cos(pi*(h + x1 - y1)) + 2*cos(pi*(h - x1 + y1)) - 6*cos(pi*(2*h + x1 + y1)) - 6*cos(pi*(x1 + y1)) - 4*cos(pi*(x1 - y1)) + 12*cos(pi*(h + x1 + y1)) + (3*h**2*pi**2*cos(pi*(2*h + x1 + y1)))/2 + (h**2*pi**2*cos(pi*(x1 + y1)))/2 - 5*h*pi*sin(pi*(2*h + x1 + y1)) + 3*pi*h*sin(pi*(x1 + y1)) + 2*h**2*pi**2*cos(pi*(x1 - y1)) + 2*pi*h*sin(pi*(h + x1 + y1)))/(h**2*pi**4),\
                                  -((cos(pi*(h + y1)) - cos(pi*y1))*(4*cos(pi*(h + x1)) - 4*cos(pi*x1) - h**2*pi**2*cos(pi*(h + x1)) + 3*pi*h*sin(pi*(h + x1)) + h*pi*sin(pi*x1)))/(h**2*pi**4),\
                                  -((cos(pi*(h + x1)) - cos(pi*x1))*(4*cos(pi*(h + y1)) - 4*cos(pi*y1) - h**2*pi**2*cos(pi*(h + y1)) + 3*pi*h*sin(pi*(h + y1)) + h*pi*sin(pi*y1)))/(h**2*pi**4),\
                                  (4*sin(pi*x1)*(cos(pi*(h + y1)) - cos(pi*y1)))/(h*pi**3) - (8*cos(pi*x1)*(cos(pi*(h + y1)) - cos(pi*y1)))/(h**2*pi**4) - (4*cos(pi*(h + x1))*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h*pi**3) + (4*sin(pi*(h + x1))*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h**2*pi**4) - (4*sin(pi*x1)*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h**2*pi**4) + (8*cos(pi*(h + x1))*(cos(pi*(h + y1)) - cos(pi*y1)))/(h**2*pi**4) + (4*sin(pi*(h + x1))*(cos(pi*(h + y1)) - cos(pi*y1)))/(h*pi**3),\
                                    (4*(sin(pi*x1) - sin(pi*(h + x1)) + pi*h*cos(pi*(h + x1)))*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h**2*pi**4),\
                                    (4*sin(pi*(h + x1))*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h**2*pi**4) - (4*cos(pi*x1)*(2*cos(pi*(h + y1)) - 2*cos(pi*y1) - h**2*pi**2*cos(pi*(h + y1)) + 2*pi*h*sin(pi*(h + y1))))/(h**2*pi**4) - (4*cos(pi*x1)*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h*pi**3) - (4*sin(pi*x1)*(sin(pi*y1) - sin(pi*(h + y1)) + pi*h*cos(pi*(h + y1))))/(h**2*pi**4) + (4*cos(pi*(h + x1))*(2*cos(pi*(h + y1)) - 2*cos(pi*y1) - h**2*pi**2*cos(pi*(h + y1)) + 2*pi*h*sin(pi*(h + y1))))/(h**2*pi**4)])
                else:
                    
                    F = np.array([(2*cos(pi*(h + x1 - y1)) + 2*cos(pi*(h - x1 + y1)) + 12*cos(pi*(x1 - h + y1)) - 6*cos(pi*(x1 - 2*h + y1)) - 6*cos(pi*(x1 + y1)) - 4*cos(pi*(x1 - y1)) + (3*h**2*pi**2*cos(pi*(x1 - 2*h + y1)))/2 + (h**2*pi**2*cos(pi*(x1 + y1)))/2 - 2*h*pi*sin(pi*(x1 - h + y1)) + 5*h*pi*sin(pi*(x1 - 2*h + y1)) - 3*pi*h*sin(pi*(x1 + y1)) + 2*h**2*pi**2*cos(pi*(x1 - y1)))/(h**2*pi**4),\
                                  -((cos(pi*y1) - cos(pi*(h - y1)))*(4*cos(pi*x1) - 4*cos(pi*(h - x1)) + h**2*pi**2*cos(pi*(h - x1)) + h*pi*sin(pi*x1) - 3*pi*h*sin(pi*(h - x1))))/(h**2*pi**4),\
                                  -((cos(pi*x1) - cos(pi*(h - x1)))*(4*cos(pi*y1) - 4*cos(pi*(h - y1)) + h**2*pi**2*cos(pi*(h - y1)) + h*pi*sin(pi*y1) - 3*pi*h*sin(pi*(h - y1))))/(h**2*pi**4),\
                                    (4*cos(pi*(h - x1))*(sin(pi*y1) + sin(pi*(h - y1)) - pi*h*cos(pi*(h - y1))))/(h*pi**3) - (4*sin(pi*(h - x1))*(sin(pi*y1) + sin(pi*(h - y1)) - pi*h*cos(pi*(h - y1))))/(h**2*pi**4) + (8*cos(pi*x1)*(cos(pi*y1) - cos(pi*(h - y1))))/(h**2*pi**4) + (4*sin(pi*x1)*(cos(pi*y1) - cos(pi*(h - y1))))/(h*pi**3) - (8*cos(pi*(h - x1))*(cos(pi*y1) - cos(pi*(h - y1))))/(h**2*pi**4) - (4*sin(pi*(h - x1))*(cos(pi*y1) - cos(pi*(h - y1))))/(h*pi**3) - (4*sin(pi*x1)*(sin(pi*y1) + sin(pi*(h - y1)) - pi*h*cos(pi*(h - y1))))/(h**2*pi**4),\
                                    (4*(sin(pi*x1) + sin(pi*(h - x1)) - pi*h*cos(pi*(h - x1)))*(sin(pi*y1) + sin(pi*(h - y1)) - pi*h*cos(pi*(h - y1))))/(h**2*pi**4),\
                                    (4*cos(pi*x1)*(2*cos(pi*y1) - 2*cos(pi*(h - y1)) + h**2*pi**2*cos(pi*(h - y1)) - 2*pi*h*sin(pi*(h - y1))))/(h**2*pi**4) - (4*sin(pi*(h - x1))*(sin(pi*y1) + sin(pi*(h - y1)) - pi*h*cos(pi*(h - y1))))/(h**2*pi**4) - (4*cos(pi*(h - x1))*(2*cos(pi*y1) - 2*cos(pi*(h - y1)) + h**2*pi**2*cos(pi*(h - y1)) - 2*pi*h*sin(pi*(h - y1))))/(h**2*pi**4) + (4*cos(pi*x1)*(sin(pi*y1) + sin(pi*(h - y1)) - pi*h*cos(pi*(h - y1))))/(h*pi**3) - (4*sin(pi*x1)*(sin(pi*y1) + sin(pi*(h - y1)) - pi*h*cos(pi*(h - y1))))/(h**2*pi**4)])
            
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
            # print('F',ElemDof)
            for k in range(nK):                
                x1 = p[s[k,0],0]
                y1 = p[s[k,0],1]                        
                if example == 1:
                    F = ForceFun1(x1,y1,h,ElemDof,k)
                else:
                    F = ForceFun2(x1,y1,h,ElemDof,k)
                for i1 in range(ElemDof):
                    i = s[k,i1]                    
                    b[i,0] = b[i,0] + F[i1]  
            
            return u, b, freeNodes
            
        A = StiffnessMatrixAssembly(self.mesh['h'],self.p,self.dofMat) # stiffness matrix
                       
        u,b,freeNodes = BoundaryCondition(self.mesh['h'],self.p,self.dofMat) # boundary condtions
                        
        u[freeNodes] = spsolve(A[freeNodes,:][:,freeNodes],b[freeNodes,0])
         
        if example == 1:
            uref = (cos(pi*self.p[:,0])- cos(pi*self.p[:,1]))/(1+pi**2)
        else:            
            uref = sin(self.p[:,0]*pi)*sin(self.p[:,1]*pi)/(2*pi**2)
                  
        if Plot_it:
            self.Plot_Solution(u,'Element ' + self.ElementType + ' FEA Solution for Elements ' ,1)
            self.Plot_Solution(uref,'Exact Solution',0)
            if self.mesh['nel']<64:
                self.Plot_Mesh()
               
        normL2  = norm((uref[freeNodes]-u[freeNodes]),ord=2)
        normLinf = norm(abs((uref[freeNodes]-u[freeNodes])),ord=np.inf)
                
        return u,normL2,normLinf
        
    def Plot_Solution(self,u,TitleStr,nn):
        Ny = int(np.sqrt(np.size(self.p,0)))
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        surf = ax.plot_surface(self.p[:,0].reshape((-1,Ny)),self.p[:,1].reshape((-1,Ny)),u.reshape((-1,Ny)),cmap=plt.cm.viridis)
        fig.colorbar(surf)
        plt.xlabel("x axis")
        plt.ylabel("y axis",rotation = "vertical",labelpad = 20)

        plt.title(TitleStr + str(np.size(self.dofMat,0)))
        if nn==0:
            plt.title(TitleStr)
        plt.ylim((np.min(self.p[:]),np.max(self.p[:])))
        plt.xlim((np.min(self.p[:]),np.max(self.p[:])))    
        plt.show()
        
    def Plot_Mesh(self):
        
        X = self.p[self.dofMat,0]
        Y = self.p[self.dofMat,1]
                
        fig, ax = plt.subplots(1,1)
        for i in range(np.size(X,0)):
            if self.ElementType == 'P1' or self.ElementType == 'P2' :
                ax.fill(X[i,0:3], Y[i,0:3], facecolor='b', edgecolor='k')
            else:
                ax.fill(X[i,0:4], Y[i,0:4], facecolor='b', edgecolor='k')
                
        ax.set_aspect('equal', 'box')
        plt.xlabel("x")
        plt.ylabel("y",rotation = "horizontal",labelpad = 20)
        plt.title('Mesh Plot of ' + self.ElementType)
        
        plt.plot(self.p[:,0],self.p[:,1],'om')
        
        plt.show()
    

def TestFunction(example,Type):
    h = 1/2**1
    
    L = 1 
    H = 1
    
    mesh = {'h':h,'L':L,'H':H}
    
    
    num = len(Type)
    meshNum = 8
    
    H = 1/2**np.arange(1,meshNum)
    L2norm = np.zeros((num,np.size(H)))
    Linfnorm = np.zeros((num,np.size(H)))
    for i in range(num):
        ElementType = Type[i]
        print(Type[i])
        for j in range(np.size(H)):
            mesh['h'] = H[j]
            F = FEA(mesh,ElementType)
            u,normL2,normLinf= F.SolveFEA(example,Plot_it=False)
            L2norm[i,j]   = normL2
            Linfnorm[i,j] = normLinf
        
        plt.figure(1)
        plt.loglog(H,L2norm[i,:],'-*')
        plt.figure(2)
        plt.loglog(H,Linfnorm[i,:],'-o')
    #%%
    LG1 = Type.copy()
    LG1.append('slope = 1')
    LG1.append('slope = 2')
    
    LG2 = Type.copy()
    LG2.append('slope = 2')
    LG2.append('slope = 3')

    
    plt.figure(1)
    plt.xlabel('Log(Element size=h)',fontsize=16)
    plt.ylabel('Log($L_{2}$ norm of Error)', fontsize=16)
    plt.title('$L_{2}$ Norm Error vs Element size', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.legend(Type,fontsize=16)
    A=np.array([10**-2,10**0])
    B=np.array([10**-4,10**-2])
    plt.loglog(A,B,'--m')
    B=np.array([10**-9,10**-5])
    plt.loglog(A,B,'--y')
    plt.legend(LG1,fontsize=16)
    
    plt.figure(2)
    plt.xlabel('Log(Element size=h)',fontsize=16)
    plt.ylabel('Log($L_{\infty}$ norm of Error)', fontsize=16)
    plt.title('$L_{\infty}$ Norm Error vs Element size', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.legend(Type,fontsize=16)
    A=np.array([10**-2,10**0])
    B=np.array([10**-7,10**-2])
    plt.loglog(A,B,'--m')
    B=np.array([10**-10,10**-4])
    plt.loglog(A,B,'--y')
    plt.legend(LG2,fontsize=16)
    
    plt.show()
        
    L = np.log(Linfnorm)
    h = np.log(H)
    a=0
    b=4
    print('slope Linf=',(L[:,b]-L[:,a])/(h[b]-h[a]))
    
    L = np.log(L2norm)
    
    print('slope L2=',(L[:,b]-L[:,a])/(h[b]-h[a]))

def ResultPlotFun(example,N):
    h = 1/2**N
    L = 1 
    H = 1
    
    mesh = {'h':h,'L':L,'H':H}
    
    Type = ['P1','P2','Q1','Q2']
        
    num = len(Type)
      
    for i in range(num):
        ElementType = Type[i]
        F = FEA(mesh,ElementType)
        _,_,_= F.SolveFEA(example,Plot_it=True)
        
#%%
example=1
# TestFunction(example,['P1','Q1'])
# TestFunction(example,['Q1','Q2'])
TestFunction(example,['P1','P2','Q1','Q2'])

N = 8
# ResultPlotFun(example,N)
