from timesteppers import StateVector
from scipy import sparse
import scipy.sparse.linalg as spla
import numpy as np


class ReactionDiffusion2D:
    
    def __init__(self, c, D, dx2, dy2):
        self.X=c
        self.t = 0
        self.iter = 0
        self.dt = None
        self.dx2 = dx2
        self.dy2 = dy2
        self.D = D
    
    def step(self, dt):
        self.dt = dt
        c = self.X
        #print(c[:,0])
        dx2 = self.dx2.matrix
        dy2 = self.dy2.matrix
        #plot_2D(c)
        #plot_2D(dx2.A)
        #plot_2D(dy2.A)
        Nx = len(c[0])
        Ny = len(c[:,0])
        
        Mx = sparse.eye(Nx, Nx)
        My = sparse.eye(Ny, Ny)
        
        spaceSteps = 1
        while spaceSteps <= 3:
            if spaceSteps % 2 == 1:
                i=0
                c_old = c
                F1 = c_old*(1-c_old)
                K1 = c_old + (dt/8)*F1
                F2 = K1*(1-K1)
                while i < Nx:
                    LHS = (Mx - (dt/8)*self.D*dx2)
                    RHS = (Mx + (dt/8)*self.D*dx2)@c_old[i] + (dt/4)*F2[i]
                    c[i] = spla.spsolve(LHS,RHS)
                    i+=1
            else:
                i=0
                c_old = c
                F1 = c_old*(1-c_old)
                K1 = c_old + (dt/4)*F1
                F2 = K1*(1-K1)
                while i < Ny:
                    LHS = (My - (dt/4)*self.D*dy2)
                    RHS = (My + (dt/4)*self.D*dy2)@c_old[:,i] + (dt/2)*F2[:,i]
                    c[:,i] = spla.spsolve(LHS,RHS)
                    i+=1
                    
            #plot_2D(c_old)
            #print(np.max(c_old))
            spaceSteps += 1
        
        self.t += dt
        self.iter += 1
        pass
    
    
class ViscousBurgers2D:
    
    def __init__(self, u, v, nu, spatial_order, domain):
        self.X = StateVector([u, v])
        self.t = 0
        self.iter = 0
        self.dt = None
    
    def step(self, dt):
        self.dt = dt
        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        #create dx2 and dy2 matrices
        dx2 = finite.DifferenceUniformGrid(2, spatial_order, domain.values[0], 0)
        dy2 = finite.DifferenceUniformGrid(2, spatial_order, domain.values[1], 1)
        
        Lx00 = -nu * dx2.matrix
        Ly00 = -nu * dy2.matrix
        L01 = Z
        L10 = Z
        
        Lx = sparse.bmat([[Lx00, L01],
                          [L10, Lx00]])
        Ly = sparse.bmat([[Ly00, L01],
                          [L10, Ly00]])
        
        
        #create dx and dy matrices
        dx = finite.DifferenceUniformGrid(1, spatial_order, domain.values[0], 0)
        dy = finite.DifferenceUniformGrid(1, spatial_order, domain.values[1], 1)
        
        f00 = lambda X: -X.variables[0]*(dx.matrix @ X.variables[0])
        f01 = lambda X: -X.variables[1]*(dy.matrix @ X.variables[0])
        f10 = lambda X: -X.variables[0]*(dx.matrix @ X.variables[1])
        f11 = lambda X: -X.variables[1]*(dy.matrix @ X.variables[1])
        self.F = sparse.bmat([[f00, f01],
                              [f10, f11]])
        
        spatialStep = 1
        while spatialStep <= 3:
            if spatialStep % 2 == 1:
                self.L = Lx
                tinydt = dt/2
            else:
                self.L = Ly
                tinydt = dt
                
            
                
            

    
class ViscousBurgers:
    
    def __init__(self, u, nu, d, d2):
        self.u = u
        self.X = StateVector([u])
        
        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu*d2.matrix
        
        f = lambda X: -X.data*(d @ X.data)
        
        self.F = f


class Wave:
    
    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        
        self.F = lambda X: 0*X.data


class SoundWave:

    def __init__(self, u, p, d, rho0, p0):
        
        self.X = StateVector([u, p])
        N = len(u)
        I = sparse.eye(N, N)
        if type(rho0)==np.ndarray:
            rho = sparse.dia_matrix((rho0, 0), shape=(N, N))
        else:
            rho = rho0 * I
        Z = sparse.csr_matrix((N, N))

        M00 = rho
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = d.matrix
        #L10
        if type(p0) == int:
            L10 = p0 * d.matrix
        else:
            L10 = Z
            for i in range (0,N):
                for j in range(0,N):
                    L10[i,j]=p0[i]*d.matrix[i,j]     
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        
        self.F = lambda X: 0*X.data


class ReactionDiffusion:
    
    def __init__(self, c, d2, c_target, D):
        
        cT=c_target
        self.X = StateVector([c])
        N = len(c)
        I = sparse.eye(N, N)
        
        if type(cT)==np.ndarray:
            cT_mat = sparse.dia_matrix((cT, 0), shape=(N, N))
        else:
            cT_mat = cT * I

        self.M = I
        self.L = -(D*d2.matrix+cT_mat)
        self.F = lambda X: -1*(X.data**2)

