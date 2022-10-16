import numpy as np
from scipy.special import factorial
from scipy import sparse
import math
import matplotlib.pyplot as plt

class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class NonUniformPeriodicGrid:

    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)


class DifferenceUniformGrid:

    def __init__(self, derivative_order, convergence_order, grid, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        #print("Check1")
        #Goal: Return D
        #initial parameters
        N=grid.N
        h=grid.dx
        ddx=derivative_order
        convE=convergence_order

        #print("Check2")
        #Create R matrix
        convE=2*math.ceil(convE/2)
        d=2*math.floor((ddx+convE-1)/2)+1
        mid=math.floor(d/2)

        R=np.zeros((d,d))
        r0=np.ones(d)
        r=h*np.arange(start=-(d-1)/2, stop=(d+1)/2, step=1)

        R[0]=r0
        i=1
        while i<d:
            R[i]=r**i
            i+=1
            
        #print("Check3")
        # Create k vector
        k=np.zeros(d)
        k[ddx]=1

        #Solve for coefficients, a
        a=math.factorial(ddx)*np.linalg.inv(R)@k
        
        #print("Check4")
        #Arrange coefficients in a sparse, derivative matrix: D
        lowerTri=np.arange(start=-N+1, stop=-N+mid+1, step=1)
        upperTri=np.arange(start=N-mid, stop=N, step=1)
        mainDiag=np.arange(start=-mid, stop=mid+1, step=1)

        #print("Main Diag: ", mainDiag)
        #print("a: ", a)
        Dmd = sparse.diags(a, mainDiag, shape=(N, N))
        DuT = sparse.diags(a[:mid], upperTri, shape=(N, N))
        DlT = sparse.diags(a[mid+1:], lowerTri, shape=(N, N))
        D = Dmd + DuT + DlT
        self.matrix = D
        pass

    def __matmul__(self, other):
        return self.matrix @ other


class DifferenceNonUniformGrid:

    def __init__(self, derivative_order, convergence_order, Grid, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.grid = Grid
        
        grid=Grid.values
        length=Grid.length
        N=Grid.N
        ddx=derivative_order
        convE=convergence_order
        
        #Create R matrix
        convE=2*math.ceil(convE/2)
        d=2*math.floor((ddx+convE-1)/2)+1
        mid=math.floor(d/2)

        # Create k vector
        k=np.zeros(d)
        k[ddx]=math.factorial(ddx)

        #Begin iteration to construct D by individual stencil
        D = sparse.csr_matrix(np.zeros((N,N)))
        i=0
        while i<N: 
            R=np.zeros((d,d))
            r0=np.ones(d)
            r=np.roll(grid, mid-i)[0:d]-grid[i]*np.ones(d)
            if i<mid:
                j=0
                while j<mid-i:
                    r[j]=r[j]-length
                    j+=1
            elif i>N-mid-1:
                j=-1 
                while j>(N-i)-mid-2:
                    r[j]=r[j]+length
                    j-=1        

            R[0]=r0
            j=1
            while j<d:
                R[j]=r**j
                j+=1
        
            Rinv=np.linalg.inv(R)

            #Solve for coefficients, a
            a=Rinv@k

            #Construct D row i, col j
            a_0s=np.concatenate((a,np.zeros(N-d)))
            a_0s_rolled=np.roll(a_0s, i-mid)
            D[i]=a_0s_rolled

            i+=1
        self.matrix = D
        pass

    def __matmul__(self, other):
        return self.matrix @ other
    
    
class Difference:

    def __matmul__(self, other):
        return self.matrix @ other


class ForwardFiniteDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [0, 1]
        diags = np.array([-1/h, 1/h])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/h
        self.matrix = matrix


class CenteredFiniteDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([-1/(2*h), 0, 1/(2*h)])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/(2*h)
        matrix[0, -1] = -1/(2*h)
        self.matrix = matrix


class CenteredFiniteSecondDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([1/h**2, -2/h**2, 1/h**2])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/h**2
        matrix[0, -1] = 1/h**2
        self.matrix = matrix


class CenteredFiniteDifference4(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-2, -1, 0, 1, 2]
        diags = np.array([1, -8, 0, 8, -1])/(12*h)
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-2, 0] = -1/(12*h)
        matrix[-1, 0] = 8/(12*h)
        matrix[-1, 1] = -1/(12*h)

        matrix[0, -2] = 1/(12*h)
        matrix[0, -1] = -8/(12*h)
        matrix[1, -1] = 1/(12*h)
        self.matrix = matrix
