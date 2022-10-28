import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.special import factorial
from collections import deque

class Timestepper:

    def __init__(self, u, f):
        self.t = 0
        self.iter = 0
        self.u = u
        self.func = f
        self.dt = None
        

    def step(self, dt):
        self.u = self._step(dt)
        self.t += dt
        self.iter += 1
        
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ForwardEuler(Timestepper):

    def _step(self, dt):
        return self.u + dt*self.func(self.u)


class LaxFriedrichs(Timestepper):

    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt*self.func(self.u)


class Leapfrog(Timestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt*self.func(self.u)
        else:
            u_temp = self.u_old + 2*dt*self.func(self.u)
            self.u_old = np.copy(self.u)
            return u_temp


class LaxWendroff(Timestepper):

    def __init__(self, u, func1, func2):
        self.t = 0
        self.iter = 0
        self.u = u
        self.f1 = func1
        self.f2 = func2

    def _step(self, dt):
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)


class Multistage(Timestepper):

    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stages = stages
        self.a = a
        self.b = b
        pass

    def _step(self, dt):
        ks = [None]*self.stages
        for i in range(0,self.stages):
            addterm = np.zeros(len(self.u))
            for j in range(0, i):
                addterm += ks[j]*self.a[i,j]
            ks[i] = self.func(self.u+(dt*addterm))
        sumks = np.zeros(len(self.u))
        for i in range(0,self.stages):
            sumks += self.b[i]*ks[i]
        return self.u + dt*sumks


class AdamsBashforth(Timestepper):

    def __init__(self, u, f, steps, dt):
        super().__init__(u, f)
        self.steps = steps
        self.dt = dt
        self.u_olds = [None]*steps

    def _step(self, dt):
        if (self.iter + 1) < self.steps:
            s = self.iter + 1
        else:
            s = self.steps
            
        for i in range(1, len(self.u_olds)):
            self.u_olds[len(self.u_olds)-i] = self.u_olds[len(self.u_olds)-i-1]
        self.u_olds[0] = self.u    
        
        aas = np.zeros(s)
        coeffs = np.zeros(s)
        for i in range(len(coeffs)):
            coeffs[i] = 1/factorial(i+1)
        
        mat = np.zeros(shape = (s,s))
        for i in range(0,s):
            for j in range(0,s):
                mat[i,j] = ((-1)**i)*(j**i)/factorial(i)
        invmat = np.linalg.inv(mat)
        aas = invmat @ coeffs
        
        nusum = np.zeros(len(self.u))
        for i in range(0,s):
            nusum += aas[i]*self.func(self.u_olds[i])
        
        return self.u + self.dt*nusum
    
    
class BackwardEuler(Timestepper):
    def __init__(self, u, L):
        super().__init__(u, L)
        N = len(u)
        self.I = sparse.eye(N, N)
    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt*self.func.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self.LU.solve(self.u)

class CrankNicolson(Timestepper):
    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        N = len(u)
        self.I = sparse.eye(N, N)
    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt/2*self.func.matrix
            self.RHS = self.I + dt/2*self.func.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self.LU.solve(self.RHS @ self.u)

class BackwardDifferentiationFormula(Timestepper):
    def __init__(self, u, L_op, steps):
        super().__init__(u, L_op)
        self.L_op = L_op
        self.steps = steps
        self.u_olds = [None]*(steps+1)
        self.delta_ts = np.zeros(steps+1)
        
    def _step(self, dt):
        #Account for initial condition where not enough previous steps exist
        if (self.iter + 1) < self.steps:
            s = self.iter + 1
        else:
            s = self.steps
        
        #Fix U_olds and create array of delta ts for previous steps
        for i in range(1, len(self.u_olds)):
            self.u_olds[len(self.u_olds)-i] = self.u_olds[len(self.u_olds)-i-1]
            self.delta_ts[len(self.delta_ts)-i] = self.delta_ts[len(self.delta_ts)-i-1]
        self.u_olds[0] = np.zeros(len(self.u))
        self.u_olds[1] = self.u
        self.delta_ts[0] = 0
        self.delta_ts[1] = dt
        
            
        aas = np.zeros(s+1)
        dts = np.zeros(s+1)
        dts[1] = 1
        
        mat = np.zeros(shape=(s+1,s+1))
        for i in range(0,s+1):
            temp_dt = 0
            for j in range(0,s+1):
                temp_dt += self.delta_ts[j]
                mat[i,j] = (temp_dt**(i))

        aas = np.linalg.inv(mat) @ dts
        
        sums = np.zeros(len(self.u))
        for i in range(1,s+1):
            sums += aas[i]*self.u_olds[i]
        
        newmat = self.L_op.matrix - a[0]*np.identity(len(self.u))
        solution = np.linalg.inv(newmat) @ sums
        solution = np.array(solution)
        solution.resize(len(self.u))
        return solution
    
class StateVector:
    
    def __init__(self, variables):
        var0 = variables[0]
        self.N = len(var0)
        size = self.N*len(variables)
        self.data = np.zeros(size)
        self.variables = variables
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[i*self.N:(i+1)*self.N], var)

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[i*self.N:(i+1)*self.N])


class IMEXTimestepper:

    def __init__(self, eq_set):
        self.t = 0
        self.iter = 0
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F
        self.dt = None

    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)

    def step(self, dt):
        self.X.data = self._step(dt)
        self.X.scatter()
        self.t += dt
        self.iter += 1


class Euler(IMEXTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        
        RHS = self.M @ self.X.data + dt*self.F(self.X)
        return self.LU.solve(RHS)


class CNAB(IMEXTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt*self.FX
            self.FX_old = self.FX
            return LU.solve(RHS)
        else:
            if dt != self.dt:
                LHS = self.M + dt/2*self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.FX - 1/2*dt*self.FX_old
            self.FX_old = self.FX
            return self.LU.solve(RHS)


class BDFExtrapolate(IMEXTimestepper):

    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        self.X_olds = [None]*(steps+1)
        self.F_olds = [None]*(steps+1)
        
        

    def _step(self, dt):
        
        #Determine steps
        if (self.iter + 1) < self.steps:
            s = self.iter + 1
        else:
            s = self.steps
        
        #Fix X_olds and F_olds
        
        Fx = self.F(self.X)
        
        for i in range(1, len(self.X_olds)):
            self.X_olds[len(self.X_olds)-i] = self.X_olds[len(self.X_olds)-i-1]
            self.F_olds[len(self.F_olds)-i] = self.F_olds[len(self.F_olds)-i-1]
        self.X_olds[0] = np.zeros(self.X.N)
        self.X_olds[1] = self.X
        self.F_olds[0] = np.zeros(self.X.N)
        self.F_olds[1] = Fx
         
        #calculate a coefficients
        ais = np.zeros(s+1)
        dts = np.zeros(s+1)
        dts[1] = 1
        
        mat = np.zeros(shape=(s+1,s+1))
        for i in range(0,s+1):
            for j in range(0,s+1):
                mat[i,j] = 1/factorial(i)*((-1*(j+1)*dt)**(i))
        ais = np.linalg.inv(mat) @ dts 
        
        #calculate b coefficients         
        bis = np.zeros(s+1)
        dt2s = np.zeros(s+1)
        dt2s[0] = 1
        
        for k in range(1,s+2):
            bis[k-1] = (-1)**(k-1)*factorial(s+1)/(factorial(s+1-k)*factorial(k))
        mat2 = np.zeros(shape=(s+1,s+1))
        for i in range(0,s+1):
            for j in range(0,s+1):
                mat2[i,j] = 1/factorial(i)*((-1*(j+1)*dt)**(i))
        print('Mat1 is: \n',mat)
        print('Mat2 is: \n', mat2)
        print('the ai matrix is: ',ais)
        print('the bi matrix is: ', bis,'\n')
        
        #compute sum of fs and sum of ais starting at 1
        suma1 = np.zeros(self.X.N)
        fsum = np.zeros(self.X.N)
        for i in range(1,s+1):
            suma1 += ais[i]*self.X_olds[i].data
            fsum += bis[i]*self.F_olds[i]
        
        #finish off the problem
        Right = self.M*ais[0]+self.L
        Left = fsum - self.M@suma1
        invR = spla.inv(Right)
        return invR @ Left