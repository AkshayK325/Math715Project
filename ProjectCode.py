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

from utillsFuncs import MeshGenerate2D


h = 0.5
L = 1 
H = 1


ElementType = 'P2'

mesh = {'h':h,'L':L,'H':H}


class FEA:
    
    def __init__(self,mesh,ElementType):
        
        self.nelx = int(mesh['L']/mesh['h'])
        self.nely = int(mesh['H']/mesh['h'])
        
        mesh['nel'] = self.nelx*self.nely
        
        if ElementType == 'P1' or ElementType == 'P2':
        
            mesh['nel'] = int(mesh['nel']*2)
            
        self.p,self.dofMat = MeshGenerate2D(mesh,ElementType) 
        
  
    
    def SolveFEA(self,bc):
        
        u=0
        
        return u
        
        
    def PlotSol(self,u):
        
        self.mesh['NodeCoordinate'] = self.mesh['NodeCoordinate'] + u.reshape((-1,2))
        
        NodesNum = (self.edofMat[:,0:-1:2]/2).astype(int)
        X = self.mesh['NodeCoordinate'][NodesNum,0]
        Y = self.mesh['NodeCoordinate'][NodesNum,1]
                
        fig, ax = plt.subplots(1,1)
        for i in range(np.size(X,0)):
            if self.mesh['ElementType']['Order']==2:
                ax.fill(X[i,0:4], Y[i,0:4], facecolor='b', edgecolor='k')
            else:
                ax.fill(X[i,:], Y[i,:], facecolor='b', edgecolor='k')
                
        ax.set_aspect('equal', 'box')
        # plt.ylim(0,self.mesh['Geometry']['H'])
        # plt.xlim(0,self.mesh['Geometry']['L'])
        plt.xlabel("x")
        plt.ylabel("y",rotation = "horizontal",labelpad = 20)
        plt.title('Deformed Mesh')
        
        plt.plot(self.mesh['NodeCoordinate'][:,0],self.mesh['NodeCoordinate'][:,1],'*m')
        
        plt.show()
        
        plt.figure()
        rotated_img = ndimage.rotate(u[0::2].reshape((self.mesh['nNodesX'],self.mesh['nNodesY'])).T, 0)
        plt.imshow(rotated_img,extent=(np.amin(self.mesh['NodeCoordinate'][:,0]), \
                                       np.amax(self.mesh['NodeCoordinate'][:,0]), \
                                        np.amin(self.mesh['NodeCoordinate'][:,1]), \
                                        np.amax(self.mesh['NodeCoordinate'][:,1])),\
                                        cmap = 'jet', origin = 'lower', interpolation='bilinear')
        plt.xlabel("x")
        plt.ylabel("y",rotation = "horizontal",labelpad = 20)
        plt.colorbar()
        plt.title('X Displacement')
        plt.show()    
        
        plt.figure()
        rotated_img = ndimage.rotate(u[1::2].reshape((self.mesh['nNodesX'],self.mesh['nNodesY'])).T, 0)
        plt.imshow(rotated_img,extent=(np.amin(self.mesh['NodeCoordinate'][:,0]), \
                                       np.amax(self.mesh['NodeCoordinate'][:,0]), \
                                        np.amin(self.mesh['NodeCoordinate'][:,1]), \
                                        np.amax(self.mesh['NodeCoordinate'][:,1])),\
                                        cmap = 'jet', origin = 'lower', interpolation='bilinear')
        plt.xlabel("x")
        plt.ylabel("y",rotation = "horizontal",labelpad = 20)
        plt.colorbar()
        plt.title('Y Displacement')
        plt.show()   
        
        pass
    
        
F = FEA(mesh,ElementType)

