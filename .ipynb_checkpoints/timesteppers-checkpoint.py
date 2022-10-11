import math
import numpy as np
import finite
import matplotlib.pyplot as plt
import scipy

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
        super().__init__(u, f)   #Sets self.t = 0
        self.stages=stages       #self.iter = 0
        self.a=a                 #self.u = u
        self.b=b                 #self.func = f
                                 #self.dt = None
        pass
    def _step(self, dt):
        N=len(self.u)
        k=np.zeros((self.stages, N))
        i=0
        while i<self.stages:
            #print("k: ", k)
            k[i]=self.func(self.u + dt*self.a[i] @ k)
            i+=1
        return self.u + dt*self.b @ k

class AdamsBashforth(Timestepper):

    def __init__(self, u, f, steps, dt):
        self.t = 0
        self.iter = 0
        self.func = f
        self.dt = dt
        self.u=np.zeros((steps,len(u)))
        self.u[0]=u
        #print("initialization: ", self.u)
        self.steps=steps
        r=np.zeros((steps,steps))
        w=np.zeros((steps,steps))
        j=1
        while j<=steps:
            b=np.zeros((j,j))
            b[0]=np.ones(j)
            i=1
            while i<j:
                b[i]=(-1)**i/math.factorial(i)*np.arange(j)**i
                i+=1
            r[j-1][:j]=scipy.special.factorial(np.arange(1,j+1))**(-1)
            w[j-1][:j]=np.linalg.inv(b)@r[j-1][:j]
            j+=1
        self.w=w
        pass
    def _step(self, dt):
        steps=self.steps
        if steps<=1:
            steps=1
            isstepsIndex=0
        else:
            isstepsIndex=1
        iters=self.iter
        #print("iteration: ", iters)
        u=self.u
        #print("a: ",type(u[0]))
        w=self.w
        N=len(u[0])

        BONUS=np.zeros(N)
        if iters>=steps:
            i=0
            while i<steps:
                BONUS=BONUS+w[steps-1,i]*self.func(u[i])
                i+=1
        else:
            i=0
            while i<=iters:
                BONUS=BONUS+w[iters,i]*self.func(u[i])
                i+=1
        u=np.roll(u,1, axis=0)
        u[0]=u[isstepsIndex]+dt*BONUS
        self.u=u
        #print("b: ", self.u[0])
        #print("c: ", type(u[0]))
        return self.u