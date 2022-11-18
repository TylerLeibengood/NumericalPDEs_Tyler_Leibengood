from timesteppers import CNRK22, CrankNicolson, StateVector, CrankNicolsonFI
from scipy import sparse
import scipy.sparse.linalg as spla
import numpy as np
import finite

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

class ReactionDiffusion:
    
    def __init__(self, c, d2, c_target, D):
        self.X = StateVector([c])
        N = len(c)
        I = sparse.eye(N,N)
        Z = sparse.csr_matrix((N,N))
        
        if type(c_target) == np.ndarray:
            ctmat = sparse.dia_matrix((c_target, 0), shape=(N,N))
        else:
            ctmat = c_target * I
            
        self.M = I
        self.L = -1*((D*d2.matrix)+ctmat)
        self.F = lambda X: -1 * (X.data**2)
        
class ReactionDiffusion1D:
    
    def __init__(self, c, d2, D, axis):
        self.X = StateVector([c], axis=axis)
        N = np.shape(c)[axis]
        I = sparse.eye(N,N)
        self.M = I
        self.L = -D*d2.matrix
        self.F = lambda X: X.data*(1-X.data)
        
class ReactionDiffusion2D:
    
    def __init__(self, c, D, dx2, dy2):
        self.X = c
        self.t = 0
        self.iter = 0
        self.dt = None
        self.dx2 = dx2
        self.dy2 = dy2
        self.D = D
    
    def step(self, dt):
        self.dt = dt/2
        sdt = dt/4
        c = self.X
        dx2 = self.dx2.matrix
        dy2 = self.dy2.matrix
        Nx = len(c[0])
        Ny = len(c[:,0])
        
        Mx = sparse.eye(Nx, Nx)
        My = sparse.eye(Ny, Ny)
        
        steps = 1
        while steps <= 3:
            if steps % 2 == 1:
                i=0
                c_old = c
                F1 = c_old*(1-c_old)
                K1 = c_old + (sdt/8)*F1
                F2 = K1*(1-K1)
                while i < Nx:
                    LHS = (Mx - (sdt/4) * self.D * dx2)
                    RHS = (Mx + (sdt/4) * self.D * dx2) @ c_old[i] + (sdt/4) * F2[i]
                    c[i] = spla.spsolve(LHS,RHS)
                    i += 1
            elif steps % 2 == 0:
                i=0
                c_old = c
                F1 = c_old*(1-c_old)
                K1 = c_old + (sdt/4)*F1
                F2 = K1*(1-K1)
                while i < Ny:
                    LHS = (My - (sdt/2) * self.D * dy2)
                    RHS = (My + (sdt/2) * self.D * dy2) @ c_old[:,i] + (sdt/2) * F2[:,i]
                    c[:,i] = spla.spsolve(LHS,RHS)
                    i += 1
                    
            steps += 1
        
        self.t += sdt
        self.iter += 1
    
    
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
        
        F00 = lambda X: -X.variables[0]*(dx.matrix @ X.variables[0])
        F01 = lambda X: -X.variables[1]*(dy.matrix @ X.variables[0])
        F10 = lambda X: -X.variables[0]*(dx.matrix @ X.variables[1])
        F11 = lambda X: -X.variables[1]*(dy.matrix @ X.variables[1])
        self.F = sparse.bmat([[F00, F01],
                              [F10, F11]])
        
        spatialStep = 1
        while spatialStep <= 3:
            if spatialStep % 2 == 1:
                self.L = Lx
                sdt = dt / 2
            elif spatialStep % 2 == 0:
                self.L = Ly
                sdt = dt



class SoundWave:

    def __init__(self, u, p, d, rho0, gammap0):
        self.X = StateVector([u, p])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))
        
        M01 = Z
        M10 = Z
        M11 = I
        
        L00 = Z
        L01 = d.matrix
        L11 = Z
        
        if (type(rho0) == np.ndarray):
            M00 = sparse.dia_matrix((rho0,0),shape=(N,N))
        else:
            M00 = rho0 * I
            
        if (type(gammap0) == np.ndarray):
            L10 = d.matrix
            for i in range(0,N):
                for j in range(0,N):
                    L10[i,j] = L10[i,j] * gammap0[i]
        else:
            L10 = gammap0 * d.matrix     


        self.M = sparse.bmat([[M00, M01],
                                  [M10, M11]])
        self.L = sparse.bmat([[L00, L01],
                                  [L10, L11]])
        self.F = lambda X: 0*X.data
        
class DiffusionBC1D:
    
    def __init__(self, c, d2, D, d1, axis):
    
        self.X = StateVector([c], axis=axis)
        N = np.shape(c)[axis]
        
        M = sparse.eye(N,N)
        self.M = M
        
        if (axis == 0):
            M = M.tocsr()
            M[0,:] = 0
            M[-1,:] = 0
            M.eliminate_zeros()
            self.M = M
        
        L = -D*d2.matrix
        self.L = L
        if (axis == 0):
            L = L.tocsr()
            L[0,:] = 0
            L[-1,:] = 0
            L[0,0] = 1 # value on left boundary is zero
            L[-1:] = d1.matrix[-1,:] # first derivative on right boundary is zero
            L.eliminate_zeros()
            self.L = L

class DiffusionBC:

    def __init__(self, c, D, spatial_order, domain):
        self.c = c
        self.XX = StateVector([c])
        self.t = 0
        self.iter = 0
        self.D = D
        
        #calculate derivatives
        xgrid, ygrid = domain.grids
        dx2 = finite.DifferenceUniformGrid(2, spatial_order, xgrid)
        dy2 = finite.DifferenceUniformGrid(2, spatial_order, ygrid)
        self.dx2 = dx2
        self.dy2 = dy2
        
        dx = finite.DifferenceUniformGrid(1, spatial_order, xgrid)
        dy = finite.DifferenceUniformGrid(1, spatial_order, ygrid)
        self.dx = dx
        self.dy = dy
        
        
    def step(self, dt):
        #take half a time step in x
        self.XX.gather()
        eqset1 = DiffusionBC1D(self.XX.data, self.dx2, self.D, self.dx, 0)
        ts1 = CrankNicolson(eqset1,0)
        self.XX.data = ts1._step(dt / 2)
        self.XX.scatter()
        
        #take a time step in y
        self.XX.gather()
        eqset2 = DiffusionBC1D(self.XX.data, self.dy2, self.D, self.dy, 1)
        ts2 = CrankNicolson(eqset2,1)
        self.XX.data = ts2._step(dt)
        self.XX.scatter()
        
        #take half a time step in x
        self.XX.gather()
        eqset3 = DiffusionBC1D(self.XX.data, self.dx2, self.D, self.dx, 0)
        ts3 = CrankNicolson(eqset3,0)
        self.XX.data = ts3._step(dt / 2)
        self.XX.scatter()
        
        #increment time and iteration
        self.t += dt
        self.iter += 1
        self.c = self.XX.data
        
        
class Wave2DBC:

    def __init__(self, u, v, p, spatial_order, domain):
        self.X = StateVector([u, v, p])
        self.iter = 0
        self.t = 0
        
        xgrid, ygrid = domain.grids
        self.dx = DifferenceUniformGrid(1, spatial_order, xgrid, axis=0)
        self.dy = DifferenceUniformGrid(1, spatial_order, ygrid, axis=1)


        def F(X):
            u = X.variables[0]
            v = X.variables[1]
            p = X.variables[2]
            dtu = -1 * (self.dx @ p)
            dtv = -1 * (self.dy @ p)
            dtp = -1 * (self.dx @ u) - (self.dy @ v)
            newvec = StateVector([dtu, dtv, dtp])
            return newvec.data
        
        self.F = F
        
        def BC(X):
            u = X.variables[0]
            u[0] = 0
            u[-1] = 0


class ReactionDiffusionFI:
    
    def __init__(self, c, D, spatial_order, grid):
        self.X = StateVector([c])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = len(c)
        I = sparse.eye(self.N)
        
        self.M = I
        self.L = -D*d2.matrix

        def F(X):
            return X.data*(1-X.data)
        self.F = F
        
        def J(X):
            c_matrix = sparse.diags(X.data)
            return sparse.eye(self.N) - 2*c_matrix
        
        self.J = J
        
class BurgersFI:
    
    def __init__(self, X, nu, spatial_order, grid):
        self.u = X
        self.X = StateVector([X])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        d = finite.DifferenceUniformGrid(1, spatial_order, grid)
        self.N = len(X)
        I = sparse.eye(self.N)
        
        self.M = I
        self.L = -nu*d2.matrix

        F = lambda X: -X.data*(d @ X.data)
        self.F = F
        
        def J(X):
            u_matrix = sparse.diags(X.data)
            I = sparse.eye(self.N)
            
            #first component: -dxu
            comp1vals = -d.matrix @ X.data
            comp1 = sparse.diags(comp1vals)
            
            #second component -u * d/du (dxu)
            comp2 = u_matrix @ (-d.matrix)
            
            return comp1 + comp2
            
        self.J = J


class ReactionTwoSpeciesDiffusion:
    
    def __init__(self, X, D, r, spatial_order, grid):
        self.X = X
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = len(X.variables[0])
        N = self.N
        M = sparse.eye(2*N, 2*N)
        self.M = M
        L01 = L10 = sparse.csr_matrix((N, N))
        L00 = L11 = -D * d2.matrix
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])
        def f(X):
            c1 = X.variables[0]
            c2 = X.variables[1]
            top = c1*(1-c1-c2)
            bottom = r*c2*(c1-c2)
            vec = StateVector([top, bottom])
            return vec.data
        
        self.F = f
        
        def J(X):
            c1 = X.variables[0]
            c1_matrix = sparse.diags(c1)
            c2 = X.variables[1]
            c2_matrix = sparse.diags(c2)
            I = sparse.eye(self.N)
            J00 = I-2*c1_matrix-c2_matrix
            J01 = -c1_matrix
            J10 = r*c2_matrix
            J11 = r*(c1_matrix-2*c2_matrix)
            return sparse.bmat([[J00, J01],
                                [J10, J11]])
        self.J = J