try:
    import casadi as ca
    import time   
    import numpy as np
    import numpy.linalg as la
    import scipy as sp
    import os
except Exception as e:
    raise e

class param_est():

    def __init__(self, state, ctrl, th, sim_th) -> None:
        self.m_cart = None # DECISON VARIABLE
        self.m_pole = None # DECISION VARIABLE
        self.l_pole = None # DECISON VARIABLE
        self.theta = None # angle of pole, rad
        self.thetadot = None # ang. Vel of pole, rad/s
        self.x = None # x direction of cart, m
        self.xdot = None # xVel of cart, m/s
        self.g = 9.8 # gravity, m/s^2
        self.dt = 0.001
        self.time_horizon = th
        self.sim_time_horizon = sim_th
        self.base_path = ""
        self.state = state
        self.ctrl = ctrl
        self.opti = None
    
    def get_state(self):
        
        # define filepaths
        state_file_path = os.path.join(self.base_path, "state.txt")

        return np.loadtxt(state_file_path, delimiter=',')


    def get_ctrl(self):
        
        # define filepaths
        ctrl_file_path = os.path.join(self.base_path, "ctrl.txt")

        return np.loadtxt(ctrl_file_path, delimiter=',')

    def get_M(self): # use current state

        M = ca.MX.zeros(2,2)
        M[0,0] = self.m_cart + self.m_pole
        M[0,1] = self.m_pole * self.l_pole * ca.cos(self.theta)
        M[1,0] = M[0,1]
        M[1,1] = self.m_pole * self.l_pole**2

        return M

    def get_C(self): # use current state

        C = ca.MX.zeros(2,2)
        C[0,1] = -self.m_pole * self.l_pole * self.thetadot * ca.sin(self.theta)

        return C

    def get_tau(self): # use current state

        tau = ca.MX.zeros(2,1)
        tau[1,0] = -self.m_pole * self.l_pole * self.g * ca.sin(self.theta)

        return tau
    
    def f(self, X, u): # use current state
        
        """
        input: State X and ctrl u, and state variables
        output: acceleration of cartpole system, qddot
        """

        # Manipulator coeffs.
        M = self.get_M()
        C = self.get_C()
        tau = self.get_tau()

        # state variables and der.
        q = ca.vertcat(X[0], X[1])
        qdot = ca.vertcat(X[2], X[3])

        # only x-dir of cart is controllable
        B = ca.vertcat(1,0)

        # acceleration of c-p system
        qddot = ca.inv(M) @ (tau + B*u - C@qdot)

        # current and next full state
        next_state = ca.vertcat(qdot, qddot)

        return next_state
    
    def rk4(self, X, U):

        """
        Runge-Kutta-4 Integration
        """
        X = X.reshape(4,1)
        k1 = self.f(X,U)
        k2 = self.f(np.add(X, (self.dt/2)*k1), U)
        k3 =  self.f(np.add(X, (self.dt/2)*k2), U)
        k4 = self.f(np.add(X, self.dt*k3), U)
        res = np.add(X, (self.dt/6)*(np.add(k1, np.add(2*k2, np.add(2*k3, k4)))))

        assert(res.shape == (4,1))

        return res 
    
    def param_estimation_leastSq_opt(self, state, ctrl): # for the entire trajectory
        
        self.opti = ca.Opti()
        self.l_pole = self.opti.variable(1)
        self.m_cart = self.opti.variable(1)
        self.m_pole = self.opti.variable(1)

        loss = 0

        for i in range(self.time_horizon-1):

            self.x = state[0,i]
            self.theta = state[1,i]
            self.xdot = state[2,i]
            self.thetadot = state[3,i]
            u = ctrl[i]

            state_ip1 = state[:,i+1]

            if ((state_ip1[0] - state[0,i]) * (state[2,i])**-1) >= self.dt*1.1 or ((state_ip1[0] - state[0,i]) * (state[2,i])**-1) < self.dt*0.8: # ignoring all huge or small time step jumps
                continue
            else:
                loss += 0.5 * ca.sumsqr(state_ip1 - self.rk4(state[:,i],u))
            
        self.opti.minimize(loss)
        self.opti.subject_to(self.l_pole > 0)
        self.opti.subject_to(self.m_cart > 0)
        self.opti.subject_to(self.m_pole > 0)
        self.opti.set_initial(self.l_pole, 0.1)
        self.opti.set_initial(self.m_cart, 0.5)
        self.opti.set_initial(self.m_pole, 0.5)

        p_opts = dict(print_time=False, verbose=False)
        s_opts = dict(print_level=0)
        self.opti.solver('ipopt',p_opts, s_opts)

    def solve(self):
        
        state = self.state # self.get_state()
        ctrl = self.ctrl # self.get_ctrl()
        self.param_estimation_leastSq_opt(state, ctrl)
        self.solution = self.opti.solve()

        #Get solutions
        pole_length = self.solution.value(self.l_pole)
        cart_mass = self.solution.value(self.m_cart)
        pole_mass = self.solution.value(self.m_pole)

        return pole_length,cart_mass, pole_mass

if __name__ == "__main__":

    # state, ctrl, th are inputs

    state = np.loadtxt("state.txt", delimiter=',')
    ctrl = np.loadtxt("ctrl.txt", delimiter=',')
    th = state.shape[1]

    pe = param_est(state, ctrl, th, 5000)    
    pole_length, cart_mass, pole_mass = pe.solve()
    print("pole_length", pole_length)
    print("cart_mass", cart_mass)
    print("pole_mass", pole_mass)
