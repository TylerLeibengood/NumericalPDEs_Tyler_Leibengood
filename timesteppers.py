import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.special import factorial
from collections import deque
from farray import axslice, apply_matrix

class StateVector:

    def __init__(self, variables, axis=0):
        self.axis = axis
        var0 = variables[0]
        shape = list(var0.shape)
        self.N = shape[axis]
        shape[axis] *= len(variables)
        self.shape = tuple(shape)
        self.data = np.zeros(shape)
        self.variables = variables
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[axslice(self.axis, i*self.N, (i+1)*self.N)], var)

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[axslice(self.axis, i*self.N, (i+1)*self.N)])


class Timestepper:

    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.X.gather()
        self.X.data = self._step(dt)
        self.X.scatter()
        self.t += dt
        self.iter += 1
    
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ExplicitTimestepper(Timestepper):

    def __init__(self, eq_set):
        super().__init__()
        self.X = eq_set.X
        self.F = eq_set.F


class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.X.data + dt*self.F(self.X)


class LaxFriedrichs(ExplicitTimestepper):

    def __init__(self, eq_set):
        super().__init__(eq_set)
        N = len(X.data)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.X.data + dt*self.F(self.X)


class Leapfrog(ExplicitTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.X_old = np.copy(self.X.data)
            return self.X.data + dt*self.F(self.X)
        else:
            X_temp = self.X_old + 2*dt*self.F(self.X)
            self.X_old = np.copy(self.X)
            return X_temp


class LaxWendroff(ExplicitTimestepper):

    def __init__(self, X, F1, F2):
        self.t = 0
        self.iter = 0
        self.X = X
        self.F1 = F1
        self.F2 = F2

    def _step(self, dt):
        return self.X.data + dt*self.F1(self.X) + dt**2/2*self.F2(self.X)


class Multistage(ExplicitTimestepper):

    def __init__(self, eq_set, stages, a, b):
        super().__init__(eq_set)
        self.stages = stages
        self.a = a
        self.b = b

        self.X_list = []
        self.K_list = []
        for i in range(self.stages):
            self.X_list.append(StateVector([np.copy(var) for var in self.X.variables]))
            self.K_list.append(np.copy(self.X.data))

    def _step(self, dt):
        X = self.X
        X_list = self.X_list
        K_list = self.K_list
        stages = self.stages

        np.copyto(X_list[0].data, X.data)
        for i in range(1, stages):
            K_list[i-1] = self.F(X_list[i-1])

            np.copyto(X_list[i].data, X.data)
            # this loop is slow -- should make K_list a 2D array
            for j in range(i):
                X_list[i].data += self.a[i, j]*dt*K_list[j]

        K_list[-1] = self.F(X_list[-1])

        # this loop is slow -- should make K_list a 2D array
        for i in range(stages):
            X.data += self.b[i]*dt*K_list[i]

        return X.data


def RK22(eq_set):
    a = np.array([[  0,   0],
                  [1/2,   0]])
    b = np.array([0, 1])
    return Multistage(eq_set, 2, a, b)


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, eq_set, steps, dt):
        super().__init__(eq_set)
        self.steps = steps
        self.dt = dt
        self.f_list = deque()
        for i in range(self.steps):
            self.f_list.append(np.copy(X.data))

    def _step(self, dt):
        f_list = self.f_list
        f_list.rotate()
        f_list[0] = self.F(self.X)
        if self.iter < self.steps:
            coeffs = self._coeffs(self.iter+1)
        else:
            coeffs = self._coeffs(self.steps)

        for i, coeff in enumerate(coeffs):
            self.X.data += self.dt*coeff*self.f_list[i].data
        return self.X.data

    def _coeffs(self, num):

        i = (1 + np.arange(num))[None, :]
        j = (1 + np.arange(num))[:, None]
        S = (-i)**(j-1)/factorial(j-1)

        b = (-1)**(j+1)/factorial(j)

        a = np.linalg.solve(S, b)
        return a


class ImplicitTimestepper(Timestepper):

    def __init__(self, eq_set, axis):
        super().__init__()
        self.axis = axis
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L

    def _LUsolve(self, data):
        if self.axis == 0:
            return self.LU.solve(data)
        elif self.axis == len(data.shape)-1:
            return self.LU.solve(data.T).T
        else:
            raise ValueError("Can only do implicit timestepping on first or last axis")


class BackwardEuler(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.M + dt*self.L
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self._LUsolve(self.X.data)


class CrankNicolson(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.M + dt/2*self.L
            self.RHS = self.M - dt/2*self.L
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self._LUsolve(apply_matrix(self.RHS, self.X.data, self.axis))


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

class CNRK22(IMEXTimestepper):
    def __init__(self, eq_set, axis):
        super().__init__(eq_set)
        self.axis = axis
    
    def _LUsolve(self, data):
        if self.axis == 0:
            return self.LU.solve(data)
        elif self.axis == len(data.shape)-1:
            return self.LU.solve(data.T).T
        else:
            raise ValueError("Can only do implicit timestepping on first or last axis")
    
    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + 0.5 * dt * self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        RHS = dt * (self.F(self.X.data + 0.5 * dt * self.F(self.X.data)) 
                     - 0.5 * self.L @ self.X.data)
        return self._LUsolve(apply_matrix(self.RHS, self.X.data, self.axis))

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

class BackwardDifferentiationFormula(Timestepper):

    def __init__(self, u, L_op, steps):
        super().__init__(u, L_op)
        self.L_op = L_op
        self.steps = steps
        self.u_olds = [None] * (steps + 1) #array of u_olds
        self.delta_ts = np.zeros(steps + 1)

    def _step(self, dt):
#         print('step: \n', self.u)
#         print('step ', self.iter)
#         print('u olds: \n', self.u_olds)
#         print('dts: \n', self.delta_ts)
        steps = self.steps
        if (self.iter+1) <= steps:
            steps = self.iter + 1
#         print('steps: ', steps)
        
        #fix the u_olds
        for i in range(1, len(self.u_olds)):
            self.u_olds[len(self.u_olds)-i] = self.u_olds[len(self.u_olds)-i-1]
            self.delta_ts[len(self.delta_ts)-i] = self.delta_ts[len(self.delta_ts)-i-1] + dt
#         for i in range(1, len(self.u_olds)-1):
#             self.delta_ts[len(self.delta_ts)-i] = self.delta_ts[len(self.delta_ts)-i-1] + dt
        self.u_olds[0] = np.zeros(len(self.u))
        self.u_olds[1] = self.u
        self.delta_ts[0] = 0
        self.delta_ts[1] = dt
        
#         print('step ', self.iter)
#         print('u olds: \n', self.u_olds)
#         print('dts: \n', self.delta_ts)
        
        ais = np.zeros(steps + 1)
        tderivs = np.zeros(steps + 1)
        tderivs[1] = 1
        matrix=np.zeros(shape=(steps+1, steps+1))
        for i in range(0, steps+1):
            for j in range(0, steps+1):
                matrix[i,j] = 1 / factorial(i) * (-1)**i * (self.delta_ts[j])**i
#         print('matrix: \n', matrix)        
        ais = np.linalg.inv(matrix) @ tderivs
#         print('ais: \n', ais)
        
        summ = np.zeros(len(self.u))
        for i in range(1, steps+1):
#             print(i)
            summ += ais[i] * self.u_olds[i]
        newmatrix = self.L_op.matrix - ais[0]*np.identity(len(self.u))
        solution = np.linalg.inv(newmatrix) @ summ
#         print('summ: \n', summ)
        solution = np.array(solution)
        solution.resize((len(self.u)))
#         print(solution)
        return solution


class BDFExtrapolate(IMEXTimestepper):

    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
#         self.X_olds = [None] * (steps + 1) #array of X_olds
#         self.F_olds = [None] * (steps + 1) #array of F_olds
        self.f_list = deque()
        self.x_list = deque()
        
        for i in range(self.steps+1):
            self.x_list.append(0)
            self.f_list.append(np.copy(self.F))
        
#         self.t = 0
#         self.iter = 0
#         self.X = eq_set.X
#         self.M = eq_set.M
#         self.L = eq_set.L
#         self.F = eq_set.F
#         self.dt = None
#         self.ais = None

    def _step(self, dt):
        
        #fix step number if needed
        steps = self.steps
        if (self.iter+1) <= steps:
            steps = self.iter + 1
            
        #compute ais
        #fix the X_olds
#         for i in range(1, len(self.X_olds)):
#             self.X_olds[len(self.X_olds)-i] = self.X_olds[len(self.X_olds)-i-1]
#         self.X_olds[0] = np.zeros(self.X.N)
#         self.X_olds[1] = self.X
        self.x_list.rotate()
        self.x_list[1] = np.copy(self.X.data)

    
    
        ais = np.zeros(steps + 1)
        tderivs = np.zeros(steps + 1)
        tderivs[1] = 1
        matrix=np.zeros(shape=(steps+1, steps+1))
        for i in range(0, steps+1):
            for j in range(0, steps+1):
                matrix[i,j] = 1 / factorial(i) * (-1)**i * (dt*j)**i        
#         print(matrix)
        ais = np.linalg.inv(matrix) @ tderivs
    
        #compute bis
        #fix the F_olds
        FX = self.F(self.X)
#         for i in range(1, len(self.F_olds)):
#             self.F_olds[len(self.F_olds)-i] = self.F_olds[len(self.F_olds)-i-1]
#         self.F_olds[0] = np.zeros(self.X.N)
#         self.F_olds[1] = self.FX
        self.f_list.rotate()
        self.f_list[1] = np.copy(FX.data)
        
#         print(self.F_olds)
        
#         bis = np.zeros(steps + 1)
#         tderivs = np.zeros(steps + 1)
#         tderivs[0] = 1
#         matrix=np.zeros(shape=(steps + 1, steps + 1))
#         for i in range(0, steps):
#             for j in range(0, steps):
#                 matrix[i,j] = 1 / factorial(i) * (-1)**i * (dt*(j))**i  
#         print(matrix)
#         bis = np.linalg.inv(matrix) @ tderivs
        bis = np.zeros(steps)
        for i in range(1,len(bis)+1):
            bis[i-1] = ((-1) ** (i-1) * factorial(steps) / 
                      (factorial(i)*factorial(steps - i)))
        
#         print('\n')
#         print('ais\n', ais)
#         print('bis\n', bis)
#         print('\n')
        
        #compute f_tilda
        f_tilda = np.zeros(self.X.N)
        for i in range(1,steps+1):
            f_tilda += bis[i-1]*self.f_list[i]
        
        #compute a sum 1
        asum1 = np.zeros(self.X.N)
        for i in range(1,steps+1):
            asum1 += ais[i]*self.x_list[i]            
        #solve for X^n
        LHS = self.M*ais[0] + self.L
        RHS = f_tilda - self.M @ asum1
#         LHS_inv = spla.inv(RHS)
        solution = spla.spsolve(LHS, RHS)
        return solution
        
        
class FullyImplicitTimestepper(Timestepper):

    def __init__(self, eq_set, tol=1e-5):
        super().__init__()
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F
        self.tol = tol
        self.J = eq_set.J
        
    def step(self, dt, guess=None):
        self.X.gather()
        self.X.data = self._step(dt, guess)
        self.X.scatter()
        self.t += dt
        self.iter += 1

        
class BackwardEulerFI(FullyImplicitTimestepper):

    def _step(self, dt, guess):
        if dt != self.dt:
            self.LHS_matrix = self.M + dt*self.L
            self.dt = dt

        RHS = self.M @ self.X.data
        if not (guess is None):
            self.X.data[:] = guess
        F = self.F(self.X)
        LHS = self.LHS_matrix @ self.X.data - dt * F
        residual = LHS - RHS
        i_loop = 0
        while np.max(np.abs(residual)) > self.tol:
            jac = self.M + dt*self.L - dt*self.J(self.X)
            dX = spla.spsolve(jac, -residual)
            self.X.data += dX
            F = self.F(self.X)
            LHS = self.LHS_matrix @ self.X.data - dt * F
            residual = LHS - RHS
            i_loop += 1
            if i_loop > 20:
                print('error: reached more than 20 iterations')
                break
        return self.X.data


class CrankNicolsonFI(FullyImplicitTimestepper):

    def _step(self, dt, guess):      
        if dt != self.dt:
            self.LHS_matrix = self.M + dt/2 * self.L
            self.RHS_matrix = self.M - dt/2 * self.L
            self.dt = dt
        F = self.F(self.X)
        RHS = self.RHS_matrix @ self.X.data + dt/2 * F
        if not (guess is None):
            self.X.data[:] = guess
        F = self.F(self.X)
        LHS = self.LHS_matrix @ self.X.data - dt/2 * F
        residual = LHS - RHS
        i_loop = 0
        while np.max(np.abs(residual)) > self.tol:
            jac = self.M + dt/2 * self.L - dt/2 * self.J(self.X)
            dX = spla.spsolve(jac, -residual)
            self.X.data += dX
            F = self.F(self.X)
            LHS = self.LHS_matrix @ self.X.data - dt/2 * F
            residual = LHS - RHS
            i_loop += 1
            if i_loop > 1000:
                print('error: reached more than 50 iterations')
                break
        return self.X.data
