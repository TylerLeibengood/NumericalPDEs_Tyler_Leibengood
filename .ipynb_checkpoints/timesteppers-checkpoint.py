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

        self.u_list = []
        self.K_list = []
        for i in range(self.stages):
            self.u_list.append(np.copy(u))
            self.K_list.append(np.copy(u))

    def _step(self, dt):
        u = self.u
        u_list = self.u_list
        K_list = self.K_list
        stages = self.stages

        np.copyto(u_list[0], u)
        for i in range(1, stages):
            K_list[i-1] = self.func(u_list[i-1])

            np.copyto(u_list[i], u)
            # this loop is slow -- should make K_list a 2D array
            for j in range(i):
                u_list[i] += self.a[i, j]*dt*K_list[j]

        K_list[-1] = self.func(u_list[-1])

        # this loop is slow -- should make K_list a 2D array
        for i in range(stages):
            u += self.b[i]*dt*K_list[i]

        return u


class AdamsBashforth(Timestepper):

    def __init__(self, u, L_op, steps, dt):
        super().__init__(u, L_op)
        self.steps = steps
        self.dt = dt
        self.f_list = deque()
        for i in range(self.steps):
            self.f_list.append(np.copy(u))

    def _step(self, dt):
        f_list = self.f_list
        f_list.rotate()
        f_list[0] = self.func(self.u)
        if self.iter < self.steps:
            coeffs = self._coeffs(self.iter+1)
        else:
            coeffs = self._coeffs(self.steps)

        for i, coeff in enumerate(coeffs):
            self.u += self.dt*coeff*self.f_list[i].data
        return self.u

    def _coeffs(self, num):

        i = (1 + np.arange(num))[None, :]
        j = (1 + np.arange(num))[:, None]
        S = (-i)**(j-1)/factorial(j-1)

        b = (-1)**(j+1)/factorial(j)

        a = np.linalg.solve(S, b)
        return a


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
        self.past = [None]*(steps+1)
        self.dts = np.zeros(steps+1)
        
    def _step(self, dt):
        #Account for initial condition where not enough previous steps exist
        if (self.iter + 1) < self.steps:
            s = self.iter + 1
        else:
            s = self.steps
        
        #Fix U_olds and create array of delta ts for previous steps
        for i in range(1, len(self.past)):
            self.past[len(self.past)-i] = self.past[len(self.past)-i-1]
            self.dts[len(self.dts)-i] = self.dts[len(self.dts)-i-1]
        self.past[0] = np.zeros(len(self.u))
        self.past[1] = self.u
        self.dts[0] = 0
        self.dts[1] = dt
        
            
        a = np.zeros(s+1)
        deets = np.zeros(s+1)
        deets[1] = 1
        
        M = np.zeros(shape=(s+1,s+1))
        for i in range(0,s+1):
            temp_dt = 0
            for j in range(0,s+1):
                temp_dt += self.dts[j]
                M[i,j] = (temp_dt**(i))

        a = np.linalg.inv(M) @ deets
        
        sums = np.zeros(len(self.u))
        for i in range(1,s+1):
            sums += a[i]*self.past[i]
        
        newmat = self.L_op.matrix - a[0]*np.identity(len(self.u))
        sol = np.linalg.inv(newmat) @ sums
        sol = np.array(sol)
        sol.resize(len(self.u))
        return sol


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
        pass

    def _step(self, dt):
        pass

