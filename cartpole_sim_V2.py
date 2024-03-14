try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf 
    import time   
    import numpy as np
    import numpy.linalg as la
    import scipy as sp
    import os
except Exception as e:
    raise e

class cartpole(): 

    """
    Dynamical model from R. Tedrake: https://underactuated.mit.edu/acrobot.html
    """

    def __init__(self, th, ctrl) -> None:
        self.m_cart = 1 # mass of cart, kg
        self.m_pole = 1 # mass of pole, kg
        self.l_pole = 0.5 # length of pole, m
        self.theta = None # angle of pole, rad
        self.thetadot = None # ang. Vel of pole, rad/s
        self.x = None # x direction of cart, m
        self.xdot = None # xVel of cart, m/s
        self.ctrl = ctrl
        self.g = 9.8 # gravity, m/s^2
        self.dt = 0.001
        self.time_horizon = th # 10000
        self.base_path = ""

    def get_M(self):

        M = np.zeros((2,2))
        M[0,0] = self.m_cart + self.m_pole
        M[0,1] = self.m_pole * self.l_pole * np.cos(self.theta)
        M[1,0] = M[0,1]
        M[1,1] = self.m_pole * self.l_pole**2

        return M

    def get_C(self):

        C = np.zeros((2,2))
        C[0,1] = -self.m_pole * self.l_pole * self.thetadot * np.sin(self.theta)

        return C

    def get_tau(self):

        tau = np.zeros((2,1))
        tau[1,0] = -self.m_pole * self.l_pole * self.g * np.sin(self.theta)

        return tau
    
    def f(self, X, u):
        
        """
        input: Manipulator Eq. Matrix Coeffs M and C, tau, ctrl u, and state variables
        output: acceleration of cartpole system, qddot
        """

        # Manipulator coeffs.
        M = self.get_M()
        C = self.get_C()
        tau = self.get_tau()
        
        # state variables and der.
        q = np.vstack([X[0], X[1]])
        qdot = np.vstack([X[2], X[3]])

        # only x-dir of cart is controllable
        B = np.vstack([1,0])

        # acceleration of c-p system
        qddot = la.inv(M) @ (tau + B*u - C@qdot) 

        # next full state
        next_state = np.vstack([qdot, qddot])

        assert(next_state.shape == (4,1))

        return next_state
    
    def euler_step(self, f):

        """
        input: current state, state_k, from dynamics of the cartpole system at time instance
        output: next state, state_k+1
        """

        state_k = f[0]
        next_state_k = f[1]

        try:
            assert(state_k.shape == (4,1))
        except Exception as e:
            print("state not the right shape")
            raise e

        state_kp1 = state_k + next_state_k*self.dt

        return state_kp1
    
    def rk4(self, X, U):

        X = X.reshape(4,1)
        k1 = self.f(X,U)
        k2 = self.f(np.add(X, (self.dt/2)*k1), U)
        k3 =  self.f(np.add(X, (self.dt/2)*k2), U)
        k4 = self.f(np.add(X, self.dt*k3), U)
        res = np.add(X, (self.dt/6)*(np.add(k1, np.add(2*k2, np.add(2*k3, k4)))))

        assert(res.shape == (4,1))

        return res 
    
    def sim(self):

        """
        this will simulate the cartpole system, where the output will be the full trajectory
        """

        # init
        X = np.zeros((4,self.time_horizon))
        U = np.zeros((1,self.time_horizon))
        u = 0.

        # initialize the system
        self.x = 0.
        self.xdot = 0.
        self.theta = np.pi
        self.thetadot = 0.

        for i in range(self.time_horizon):

            # ctrl
            u = self.ctrl
            U[0,i] = u[i]

            # "append" trajectory at instance
            X[0,i] = self.x
            X[1,i] = self.theta
            X[2,i] = self.xdot
            X[3,i] = self.thetadot

            # rk4 integration
            state_next = self.rk4(X[:,i], u[i])  

            # update state
            self.x = state_next[0][0]
            self.theta = state_next[1][0]
            self.xdot = state_next[2][0]
            self.thetadot = state_next[3][0]
        
        return X, U

    
    def store_to_txt_file(self, state, ctrl):

        """
        stores to text file in filepath
        """
        
        # define filepaths
        state_file_path = os.path.join(self.base_path, "state.txt")
        ctrl_file_path = os.path.join(self.base_path, "ctrl.txt")

        # save textfiles in filepath
        np.savetxt(state_file_path, state, delimiter=',')
        np.savetxt(ctrl_file_path, ctrl, delimiter=',')

if __name__ == "__main__":

    cp = cartpole(theta_init=3.7*np.pi/4, th=5000)
    X, U = cp.sim()
    cp.store_to_txt_file(X, U)