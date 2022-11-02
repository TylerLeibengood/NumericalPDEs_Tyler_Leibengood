from timesteppers import StateVector
from scipy import sparse
import numpy as np

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
        self.L = -(D*d2+cT_mat)
        self.F = lambda X: -1*(X.data**2)

