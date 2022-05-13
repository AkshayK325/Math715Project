import numpy as np



def MeshGenerate2D(mesh,Type):       
    
    nelx = int(mesh['L']/mesh['h'])
    nely = int(mesh['H']/mesh['h'])
    
    if Type == 'Q1':
        
        mesh['dofPerElem'] = 4
        
        edofMat = np.zeros((mesh['nel'],mesh['dofPerElem']))

        nNodex = nelx+1
        nNodey = nely+1
        
        for elx in range(nelx):
            for ely in range(nely):
                el = ely+elx*nely
                n1=(nely+1)*elx+ely
                n2=(nely+1)*(elx+1)+ely
                edofMat[el,:]=np.array([n1,n2,n2+1,n1+1]);
                
    elif Type == 'P1':
        mesh['dofPerElem'] = 3
        
        edofMat = np.zeros((mesh['nel'],mesh['dofPerElem']))
        
        nNodex = nelx+1
        nNodey = nely+1
        
        el=0
        for elx in range(nelx):
            for ely in range(nely):
                
                n1=(nely+1)*elx+ely
                n2=(nely+1)*(elx+1)+ely
                
                edofMat[el,:]=np.array([n1,n2,n1+1]);
                el = el + 1    
                edofMat[el,:]=np.array([n2+1,n1+1,n2]);
                el = el + 1  
                
    elif Type == 'Q2':
        mesh['dofPerElem'] = 9
        
        edofMat = np.zeros((mesh['nel'],mesh['dofPerElem']))
        
        nNodex = 2*nelx+1
        nNodey = 2*nely+1
    
        for elx in range(nelx):
            for ely in range(nely):
                el = ely+elx*nely
                n1=(2*nely+1)*2*elx+ely*2
                n2=(2*nely+1)*(2*elx+1)+ely*2
                n3=(2*nely+1)*(2*elx+2)+ely*2
                        
                edofMat[el,:]=np.array([n1,n3,n3+2,n1+2,n2,n3+1,\
                                        n2+2,n1+1,n2+1]);
    
    elif Type == 'P2':
        mesh['dofPerElem'] = 6
        
        edofMat = np.zeros((mesh['nel'],mesh['dofPerElem']))
        
        nNodex = 2*nelx+1
        nNodey = 2*nely+1
        
        el=0
        for elx in range(nelx):
            for ely in range(nely):
                n1=(2*nely+1)*2*elx+ely*2
                n2=(2*nely+1)*(2*elx+1)+ely*2
                n3=(2*nely+1)*(2*elx+2)+ely*2
        
                edofMat[el,:]=np.array([n1,n3,n1+2,n2,n2+1,n1+1]);
                
                el = el + 1    
                
                edofMat[el,:]=np.array([n3+2,n1+2,n3,n2+2,n2+1,n3+1]);
                # edofMat[el,:]=np.array([n3+2,n3,n1+2,n3+1,n2+1,n2+2]);

                
                el = el + 1  

    
    x,y = np.meshgrid(np.arange(nNodex),np.arange(nNodey),indexing='ij')
  
    p=np.zeros((np.size(x),2))
        
    p[:,0]=x.reshape(-1)/(nNodex-1)
    p[:,1]=y.reshape(-1)/(nNodey-1)    
    
    return p,edofMat.astype(int),mesh
    