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
except Exception as e:
    raise e

class fisherComputation():

    def __init__(self, state, ctrl, th, fisher_prev) -> None:
        
        # self.m_cart = 1. # mass of cart, kg
        # self.m_pole = 1. # mass of pole, kg
        # self.l_pole = 0.5 # length of pole, m
        self.theta = None # angle of pole, rad
        self.thetadot = None # ang. Vel of pole, rad/s
        self.x = None # x direction of cart, m
        self.xdot = None # xVel of cart, m/s
        self.g = 9.8 # gravity, m/s^2
        self.dt = 0.01
        self.state = state
        self.ctrl = ctrl
        self.time_horizon = th
        self.fisher_prev = fisher_prev
        self.base_path = ""
    
    def get_M(self, l_pole, m_cart, m_pole):

        # M = np.zeros((2,2))
        # M[0,0] = self.m_cart + m_pole
        # M[0,1] = m_pole * l_pole * np.cos(self.theta[0])
        # M[1,0] = M[0,1]
        # M[1,1] = m_pole * l_pole**2

        M = jnp.array([[m_cart + m_pole, m_pole * l_pole * np.cos(self.theta[0])],
                      [m_pole * l_pole * np.cos(self.theta[0]), m_pole * l_pole**2]])
        test = jla.det(M)
        return test #M

    def get_C(self, l_pole, m_cart, m_pole):

        # C = np.zeros((2,2))
        # C[0,1] = -m_pole * l_pole * self.thetadot[0] * np.sin(self.theta[0])

        C = jnp.array([[0, -m_pole * l_pole * self.thetadot[0] * np.sin(self.theta[0])],
                       [0, 0]])
        test = jla.det(C)
        return test # C

    def get_tau(self, l_pole, m_cart, m_pole):

        # tau = np.zeros((2,1))
        # tau[1,0] = -m_pole * l_pole * self.g * np.sin(self.theta[0])

        tau = jnp.array([[0],
                         [-m_pole * l_pole * self.g * np.sin(self.theta[0])]])

        test = jla.det(tau)
        return test # tau
    
    def logged_loss(self, l_pole, m_cart, m_pole, u):
        
        """
        input: Manipulator Eq. Matrix Coeffs M and C, tau, ctrl u, and state variables
        output: acceleration of cartpole system, qddot
        """

        # state variables and der.
        q = jnp.vstack([self.x[0], self.theta[0]])
        qdot = jnp.vstack([self.xdot[0], self.thetadot[0]])

        # only x-dir of cart is controllable
        B = jnp.vstack([1,0])

        # Manipulator Eqs.

        M = jnp.array([[m_cart + m_pole, m_pole * l_pole * np.cos(self.theta[0])],
                      [m_pole * l_pole * np.cos(self.theta[0]), m_pole * l_pole**2]])
        
        C = jnp.array([[0, -m_pole * l_pole * self.thetadot[0] * np.sin(self.theta[0])],
                       [0, 0]])
        
        tau = jnp.array([[0],
                         [-m_pole * l_pole * self.g * np.sin(self.theta[0])]])

        # acceleration of c-p system
        qddot = jla.inv(M) @ (tau + B*u[0] - C@qdot)

        # logged loss
        state_dot = jnp.vstack([qdot, qddot])

        state_next = jnp.vstack([self.x[1], self.theta[1], self.xdot[1], self.thetadot[1]])

        # logged_loss = jnp.log(jnp.exp(0.5 * (state_next - state_dot)**2))
        # logged_loss = jnp.exp(0.5 * (state_next - state_dot)**2)
        logged_loss = 0.5 * (state_next - state_dot)**2
        logged_loss = jnp.sum(logged_loss, axis=0).reshape((1,1))

        return jnp.squeeze(logged_loss)

    
    def get_psi(self, l_pole_est, m_cart_est, m_pole_est, u):

        config.update("jax_debug_nans", True)

        # dM = jax.grad(self.get_M, argnums=0)(l_pole_est, m_cart_est, m_pole_est)
        # dC = jax.grad(self.get_C, argnums=0)(l_pole_est, m_cart_est, m_pole_est)
        # dtau = jax.grad(self.get_tau, argnums=0)(l_pole_est, m_cart_est, m_pole_est)
        
        dfdl_pole = jax.grad(self.logged_loss, argnums=0)(l_pole_est, m_cart_est, m_pole_est, u)
        dfdm_cart = jax.grad(self.logged_loss, argnums=1)(l_pole_est, m_cart_est,m_pole_est, u)
        dfdm_pole = jax.grad(self.logged_loss, argnums=2)(l_pole_est, m_cart_est,m_pole_est, u)
        
        dfdl_pole_sol = dfdl_pole#(l_pole_est, m_pole_est, u)
        dfdm_cart_sol = dfdm_cart#(l_pole_est, m_pole_est, u)
        dfdm_pole_sol = dfdm_pole

        psi = np.array([dfdl_pole_sol, dfdm_cart_sol, dfdm_pole_sol])
        
        return psi

    def get_fisher(self, psi_list):

        fisher_sum = self.fisher_prev
        
        for t in range(len(psi_list)):
            
            psi_list_t = psi_list[t].reshape(3,1)
            psi_sq = psi_list_t @ psi_list_t.T

            for i in range(3):
                for j in range(3):
                    fisher_sum[i,j] += psi_sq[i,j]
        
        # assert(np.array(fisher_sum) >= 0)

        return fisher_sum

    def get_state(self):
        
        # define filepaths
        state_file_path = os.path.join(self.base_path, "state.txt")

        return np.loadtxt(state_file_path, delimiter=',')


    def get_ctrl(self):
        
        # define filepaths
        ctrl_file_path = os.path.join(self.base_path, "ctrl.txt")

        return np.loadtxt(ctrl_file_path, delimiter=',')
    
    def run(self, l_pole_est, m_cart_est, m_pole_est):

        state = self.state # self.get_state()
        ctrl = self.ctrl # self.get_ctrl()

        psi_list = []

        for i in range(self.time_horizon-1):

            self.x = state[0,i:i+2]
            self.xdot = state[1,i:i+2]
            self.theta = state[2,i:i+2]
            self.thetadot = state[3,i:i+2]

            u = ctrl[i:i+2]

            psi = self.get_psi(l_pole_est, m_cart_est, m_pole_est, u)
            # print("psi: ", psi)
            psi_list.append(psi)

        fisher = self.get_fisher(psi_list)
        
        return fisher

        
if __name__ == "__main__":

    fisher = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])

    cf = fisherComputation(np.loadtxt(os.path.join("", "state.txt"), delimiter=','), 
                           np.loadtxt(os.path.join("", "ctrl.txt"), delimiter=','), 
                           200, #np.loadtxt(os.path.join("", "state.txt"), delimiter=',').shape[1],
                           fisher)
    print(cf.run(0.5040119640371667, 10., 10.))