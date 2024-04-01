try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf 
    import time   
    import numpy as np
    import numpy.linalg as la
    import scipy as sp
    import os
    from inits import inits
except Exception as e:
    raise e

class cartpole(inits):

    """
    Dynamical model from R. Tedrake: https://underactuated.mit.edu/acrobot.html
    """

    def __init__(self, ctrl) -> None:
        super().__init__()

        """
        physical parameters of the cartpole
        """
        self.m_cart = self.vars['m_cart']               # mass of cart, kg
        self.m_pole = self.vars['m_pole']               # mass of pole, kg
        self.l_pole = self.vars['l_pole']               # length of pole, m

        """
        state variables
        """
        self.theta = None                               # angle of pole, rad
        self.thetadot = None                            # ang. Vel of pole, rad/s
        self.x = None                                   # x direction of cart, m
        self.xdot = None                                # xVel of cart, m/s

        """
        initial conditions
        """
        self.x_init = self.IC['x']                      # initial x
        self.theta_init = self.IC['theta']              # initial theta
        self.xdot_init = self.IC['xdot']                # initial xdot 
        self.thetadot_init = self.IC['thetadot']        # initial thetadot

        """
        control inputs at every time iteration
        """
        self.ctrl = ctrl                                # ctrl inputs (array type)

        """
        other constants
        """
        self.g = self.vars['gravity']                   # gravity, m/s^2
        self.dt = self.vars['timestep']                 # timestep
        self.time_horizon = self.vars['time_horizon']   # time horizon

    def get_M(self):

        """
        Mass matrix
        """

        M = np.zeros((2,2))
        M[0,0] = self.m_cart + self.m_pole
        M[0,1] = self.m_pole * self.l_pole * np.cos(self.theta)
        M[1,0] = M[0,1]
        M[1,1] = self.m_pole * self.l_pole**2

        return M

    def get_C(self):

        """
        Coriolis matrix
        """

        C = np.zeros((2,2))
        C[0,1] = -self.m_pole * self.l_pole * self.thetadot * np.sin(self.theta)

        return C

    def get_tau(self):

        """
        Conservative forces
        """

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
        qdot = np.vstack([X[2], X[3]])

        # only x-dir of cart is controllable
        B = np.vstack([1,0])

        # acceleration of c-p system
        qddot = la.inv(M) @ (tau + B*u - C@qdot) 

        # next full state
        next_state = np.vstack([qdot, qddot])

        assert(next_state.shape == (4,1))

        return next_state
    
    def rk4(self, X, U):

        """
        Runge-Kutta integrator
        """

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
        u = 0.

        # initialize the system
        self.x = self.x_init
        self.xdot = self.xdot_init
        self.theta = self.theta_init
        self.thetadot = self.thetadot_init

        for i in range(self.time_horizon):

            # ctrl
            u = self.ctrl

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
        
        return X