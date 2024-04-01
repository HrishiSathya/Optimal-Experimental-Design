try:
    import jax
    import jax.numpy as jnp
    import jax.numpy.linalg as jla
    from jax.config import config
    import time   
    import numpy as np
    import numpy.linalg as la
    import scipy as sp
    import os
    from inits import inits
except Exception as e:
    raise e

class fisherComputation(inits):

    def __init__(self, state, ctrl, m_pole, l_pole, m_cart, fisher_prev, output_variance) -> None:
        super().__init__()

        """
        parameter estimates
        """
        self.m_cart = m_cart                            # mass of cart, kg
        self.m_pole = m_pole                            # mass of pole, kg
        self.l_pole = l_pole                            # length of pole, m

        """
        state variables
        """
        self.theta = None                               # angle of pole, rad
        self.thetadot = None                            # ang. Vel of pole, rad/s
        self.x = None                                   # x direction of cart, m
        self.xdot = None                                # xVel of cart, m/s
        self.state = state                              # trajectories
        self.ctrl = ctrl                                # control inputs

        """
        other constants
        """
        self.g = self.vars['gravity']                   # gravity, m/s^2
        self.dt = self.vars['timestep']                 # timstep
        self.time_horizon = self.vars['time_horizon']   # time horizon
        self.fisher_prev = fisher_prev                  # previous fisher information
        self.variance = output_variance
    
    def logged_measurement_model(self, l_pole, m_cart, m_pole, u):
        
        """
        input: ctrl u, and state variables
        output: acceleration of cartpole system, qddot
        """

        # state variables and der.
        qdot = jnp.vstack([self.xdot[0], self.thetadot[0]])

        # only x-dir of cart is controllable
        B = jnp.vstack([1,0])

        # Manipulator consts

        M = jnp.array([[m_cart + m_pole, m_pole * l_pole * np.cos(self.theta[0])],
                      [m_pole * l_pole * np.cos(self.theta[0]), m_pole * l_pole**2]])
        
        C = jnp.array([[0, -m_pole * l_pole * self.thetadot[0] * np.sin(self.theta[0])],
                       [0, 0]])
        
        tau = jnp.array([[0],
                         [-m_pole * l_pole * self.g * np.sin(self.theta[0])]])

        # acceleration of c-p system
        qddot = jla.inv(M) @ (tau + B*u[0] - C@qdot)

        # next state
        state_dot = jnp.vstack([qdot, qddot])
        state_next = jnp.vstack([self.x[1], self.theta[1], self.xdot[1], self.thetadot[1]])

        # define measurement model (loss function of the parameter estimation problem)
        logged_measurement_model = -0.5 * (state_next - state_dot)**2
        logged_measurement_model = jnp.sum(logged_measurement_model, axis=0).reshape((1,1))

        return jnp.squeeze(logged_measurement_model)

    
    def get_psi(self, l_pole_est, m_cart_est, m_pole_est, u):

        """
        obtain the psi value at time instance
        """

        config.update("jax_debug_nans", True)
        
        dfdl_pole = jax.grad(self.logged_measurement_model, argnums=0)(l_pole_est, m_cart_est, m_pole_est, u)
        dfdm_cart = jax.grad(self.logged_measurement_model, argnums=1)(l_pole_est, m_cart_est,m_pole_est, u)
        dfdm_pole = jax.grad(self.logged_measurement_model, argnums=2)(l_pole_est, m_cart_est,m_pole_est, u)

        psi = np.array([dfdl_pole, dfdm_cart, dfdm_pole])
        
        return psi

    def get_fisher(self, psi_list):

        """
        Accumulate all the psi . psi^T for all time iterations 
        """

        fisher_sum = self.fisher_prev
        
        for t in range(len(psi_list)):
            
            psi_list_t = psi_list[t].reshape(3,1)
            psi_sq = psi_list_t @ np.linalg.inv(self.variance*np.eye(psi_list_t.shape[1])) @ psi_list_t.T

            for i in range(3):
                for j in range(3):
                    if i == j:
                        fisher_sum[i,j] += psi_sq[i,j]

        return fisher_sum
    
    def run(self):

        """
        Compute the fisher information based on estimates
        """

        l_pole_est = self.l_pole
        m_cart_est = self.m_cart
        m_pole_est = self.m_pole
        state = self.state
        ctrl = self.ctrl
        psi_list = []

        for i in range(self.time_horizon-1):

            self.x = state[0,i:i+2]
            self.xdot = state[1,i:i+2]
            self.theta = state[2,i:i+2]
            self.thetadot = state[3,i:i+2]
            u = ctrl[i:i+2]

            psi = self.get_psi(l_pole_est, m_cart_est, m_pole_est, u)
            psi_list.append(psi)

        fisher = self.get_fisher(psi_list)
        
        return fisher