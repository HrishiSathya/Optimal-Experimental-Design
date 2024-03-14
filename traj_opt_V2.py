try:
    import jax
    import jax.numpy as jnp
    import jax.numpy.linalg as jla
    from jax.config import config
    import casadi
    import time   
    import tqdm
    import numpy as np
    import numpy.linalg as la
    import scipy as sp
    import os
except Exception as e:
    raise e

class Traj_Optimization():
    
    def __init__(self, l_pole_est, m_cart_est, m_pole_est, th, fisher_prev, x_init, theta_init, xdot_init, thetadot_init, ctrl_init, conv):

        self.m_cart = m_cart_est # mass of cart, kg
        self.m_pole = m_pole_est # mass of pole, kg
        self.l_pole = l_pole_est # length of pole, m
        self.theta = None # angle of pole, rad
        self.thetadot = None # ang. Vel of pole, rad/s
        self.x = None # x direction of cart, m
        self.xdot = None # xVel of cart, m/s
        self.x_init = x_init
        self.theta_init = theta_init
        self.xdot_init = xdot_init
        self.thetadot_init = thetadot_init
        self.ctrl_init = ctrl_init
        self.g = 9.8 # gravity, m/s^2
        self.dt = 0.001
        self.weight_1 = None
        self.weight_2 = None
        self.freq_1 = None
        self.freq_2 = None
        self.state = None
        self.ctrl = None
        self.fisher_prev = fisher_prev
        self.time_horizon = th
        self.opti = None
        self.solution = None
        self.base_path = ""
        self.conv = conv

    def get_M(self,X):

        X = X.reshape((4,1))

        M = casadi.MX.zeros(2,2)
        M[0,0] = self.m_cart + self.m_pole
        M[0,1] = self.m_pole * self.l_pole * casadi.cos(X[1])
        M[1,0] = M[0,1]
        M[1,1] = self.m_pole * self.l_pole**2

        return M

    def get_C(self, X):

        X = X.reshape((4,1))

        C = casadi.MX.zeros(2,2)
        C[0,1] = -self.m_pole * self.l_pole * X[3] * casadi.sin(X[1])

        return C

    def get_tau(self, X):

        X = X.reshape((4,1))

        tau = casadi.MX.zeros(2,1)
        tau[1,0] = -self.m_pole * self.l_pole * self.g * casadi.sin(X[1])

        return tau
    
    def f(self, X, u): # use current state
        
        """
        input: State X and ctrl u, and state variables
        output: acceleration of cartpole system, qddot
        """

        # Manipulator coeffs.
        M = self.get_M(X)
        C = self.get_C(X)
        tau = self.get_tau(X)

        # state variables and der.
        q = casadi.vertcat(X[0], X[1])
        qdot = casadi.vertcat(X[2], X[3])

        # only x-dir of cart is controllable
        B = casadi.vertcat(1,0)

        # acceleration of c-p system
        qddot = casadi.inv(M) @ (tau + B*u - C@qdot)

        # current and next full state
        next_state = casadi.vertcat(qdot, qddot)

        return next_state
    
    def rk4(self, X, U):

        """
        Runge-Kutta-4 Integration
        """
        X = X.reshape((4,1))
        k1 = self.f(X,U)
        k2 = self.f(np.add(X, (self.dt/2)*k1), U)
        k3 =  self.f(np.add(X, (self.dt/2)*k2), U)
        k4 = self.f(np.add(X, self.dt*k3), U)
        res = np.add(X, (self.dt/6)*(np.add(k1, np.add(2*k2, np.add(2*k3, k4)))))

        assert(res.shape == (4,1))

        return res 
    
    def get_psi(self, state_k, state_kp1, u):

        state_k = state_k.reshape((4,1))
        state_kp1 = state_kp1.reshape((4,1))
        u = u.reshape((1,1))

        # Coefficients
        M = self.get_M(state_k)
        C = self.get_C(state_k)
        tau = self.get_tau(state_k)

        """
        ctrl'bility
        """
        # only x-dir of cart is controllable
        B = casadi.vertcat(1,0)

        """
        state variables for qdot
        """
        q = state_k[:2]
        qdot = state_k[2:]
        qddot = casadi.inv(M) @ (tau + B*u - C@qdot)

        """
        outer component of chain rule
        """

        dLdtheta = 2*(state_kp1 - casadi.vertcat(qdot, qddot)).reshape((4,1))

        """
        dfdl_pole
        """
        
        dMdl_pole = casadi.MX.zeros(2,2)
        dCdl_pole = casadi.MX.zeros(2,2)
        dtaudl_pole = casadi.MX.zeros(2,1)
        dresdl_pole = None

        # Mass partial computation
        dMdl_pole[0,1] = self.m_pole * casadi.cos(state_k[1])
        dMdl_pole[1,0] = dMdl_pole[0,1]
        dMdl_pole[1,1] = 2 * self.m_pole * self.l_pole

        # Coriolis partial computation
        dCdl_pole[0,1] = -self.m_pole * state_k[3] * casadi.sin(state_k[1])

        # Conserv. Forces partial computation
        dtaudl_pole[1] = -self.m_pole * self.g * casadi.sin(state_k[1])

        dresdl_pole = casadi.vertcat(dLdtheta[:2]*casadi.MX.zeros(2,1),
                                     dLdtheta[2:]*(-(-casadi.inv(M**2) @ dMdl_pole @ tau + casadi.inv(M) @ dtaudl_pole - casadi.inv(M**2) @ dMdl_pole @ B*u - casadi.inv(M) @ dCdl_pole@qdot + casadi.inv(M**2) @ dMdl_pole @ C@qdot)))
        
        assert(dresdl_pole.shape == (4,1))
        dresdl_pole = casadi.sum1(dresdl_pole).reshape((1,1))
        assert(dresdl_pole.shape == (1,1))

        """
        dfdm_cart
        """
        dMdm_cart = casadi.MX.zeros(2,2)
        dCdm_cart = casadi.MX.zeros(2,2)
        dtaudm_cart = casadi.MX.zeros(2,1)
        dresdm_cart = None

        # Mass partial computation
        dMdm_cart[0,0] = 1.

        dresdm_cart = casadi.vertcat(dLdtheta[:2]*casadi.MX.zeros(2,1),
                                     dLdtheta[2:]*(-(-casadi.inv(M**2) @ dMdm_cart @ tau + casadi.inv(M) @ dtaudm_cart - casadi.inv(M**2) @ dMdm_cart @ B*u - casadi.inv(M) @ dCdm_cart@qdot + casadi.inv(M**2) @ dMdm_cart @ C@qdot)))

        assert(dresdm_cart.shape == (4,1))
        dresdm_cart = casadi.sum1(dresdm_cart).reshape((1,1))
        assert(dresdm_cart.shape == (1,1))

        """
        dfdm_pole
        """
        dMdm_pole = casadi.MX.zeros(2,2)
        dCdm_pole = casadi.MX.zeros(2,2)
        dtaudm_pole = casadi.MX.zeros(2,1)
        dresdm_pole = None

        # Mass partial computation
        dMdm_pole[0,0] = 1.
        dMdm_pole[0,1] = self.l_pole * casadi.cos(state_k[1])
        dMdm_pole[1,0] = dMdm_pole[0,1]
        dMdm_pole[1,1] = self.l_pole**2

        # Coriolis partial computation
        dCdm_pole[0,1] = -self.l_pole * state_k[3] * casadi.sin(state_k[1])

        # Conserv. Forces partial computation
        dtaudm_pole[1] = -self.l_pole * self.g * casadi.sin(state_k[1])

        dresdm_pole = casadi.vertcat(dLdtheta[:2]*casadi.MX.zeros(2,1),
                                      dLdtheta[2:]*(-(-casadi.inv(M**2) @ dMdm_pole @ tau + casadi.inv(M) @ dtaudm_pole - casadi.inv(M**2) @ dMdm_pole @ B*u - casadi.inv(M) @ dCdm_pole@qdot + casadi.inv(M**2) @ dMdm_pole @ C@qdot)))
        
        assert(dresdm_pole.shape == (4,1))
        dresdm_pole = casadi.sum1(dresdm_pole).reshape((1,1))
        assert(dresdm_pole.shape == (1,1))
        
        psi = casadi.MX.zeros(1,3)
        psi[0,0] = dresdl_pole 
        psi[0,1] = dresdm_cart 
        psi[0,2] = dresdm_pole
        
        return psi

    def summation(self, psi_list):

        fisher_sum = self.fisher_prev
        
        for t in range(len(psi_list)):
            
            psi_list_t = psi_list[t].reshape((3,1))
            psi_sq = psi_list_t @ psi_list_t.T

            for i in range(3):
                for j in range(3):
                    fisher_sum[i,j] += psi_sq[i,j]
        
        return fisher_sum
    
    def get_fisher(self, state, ctrl):

        psi_list = []

        for i in range(self.time_horizon-1):

            psi = self.get_psi(state[:,i],state[:,i+1],ctrl[i])
            psi_list.append(psi)

        fisher = self.summation(psi_list)
        
        return fisher
    
    def D_opti(self):

        """
        set up the optimization
        """
        self.opti = casadi.Opti()

        """
        decision variables
        """
        self.state = self.opti.variable(4,self.time_horizon)
        self.ctrl = self.opti.variable(1,self.time_horizon)
        self.weight_1 = self.opti.variable(1)
        self.weight_2 = self.opti.variable(1)
        self.freq_1 = self.opti.variable(1)
        self.freq_2 = self.opti.variable(1)

        """
        objective minimization
        """
        loss = 0 
        fisher_k = self.get_fisher(self.state,self.ctrl)
        loss -= (fisher_k[0,0] + fisher_k[1,1] + fisher_k[2,2])

        for i in range(self.time_horizon-1):

            """
            ctrl bounds
            """
            self.opti.subject_to(self.weight_1 <= 10.)
            self.opti.subject_to(self.weight_1 >= -10.)
            self.opti.subject_to(self.weight_2 <= 10.)
            self.opti.subject_to(self.weight_2 >= -10.)
            self.opti.subject_to(self.freq_1 <= 10.)
            self.opti.subject_to(self.freq_1 >= -10.)
            self.opti.subject_to(self.freq_2 <= 10.)
            self.opti.subject_to(self.freq_2 >= -10.)
            self.opti.subject_to(self.ctrl[i+1] == self.weight_1*casadi.sin(self.freq_1*i) + self.weight_2*casadi.cos(self.freq_2*i))
            
            """
            next state
            """
            state_ipl = self.state[:,i+1]
            
            """
            dynamics and kinematics constraints
            """
            self.opti.subject_to(state_ipl == self.rk4(self.state[:,i],self.ctrl[i]))

            for k in range(2):
                loss += (self.state[k,i+1] - self.state[k,i])/self.dt - self.state[k+2,i]
        
        # set the ICs
        self.opti.subject_to(self.state[0,0] == self.x_init) #self.x_init
        self.opti.subject_to(self.state[1,0] == self.theta_init) #self.theta_init
        self.opti.subject_to(self.state[2,0] == self.xdot_init) #self.xdot_init
        self.opti.subject_to(self.state[3,0] == self.thetadot_init) #self.thetadot_init
        self.opti.subject_to(self.ctrl[0] == self.ctrl_init) #self.ctrl_init
        
        self.opti.minimize(loss)

        """
        solver
        """
        p_opts = dict(print_time=False, verbose=False)
        s_opts = dict(print_level=0, acceptable_tol=self.conv, max_iter = 15000, linear_solver='spral')
        self.opti.solver('ipopt',p_opts, s_opts)
    
    def run(self):

        self.D_opti()
        self.solution = self.opti.solve()
        u = self.solution.value(self.ctrl)
        x = self.solution.value(self.state)
        fisher_k = self.solution.value(self.get_fisher(self.state, self.ctrl))
        w1 = self.solution.value(self.weight_1)
        w2 = self.solution.value(self.weight_2)
        f1 = self.solution.value(self.freq_1)
        f2 = self.solution.value(self.freq_2)
        
        # det_fisher_k = fisher_k[0,0]*(fisher_k[1,1]*fisher_k[2,2] - fisher_k[2,1]*fisher_k[1,2]) - fisher_k[0,1]*(fisher_k[1,0]*fisher_k[2,2] - fisher_k[1,2]*fisher_k[2,0]) + fisher_k[0,2]*(fisher_k[1,0]*fisher_k[2,1] - fisher_k[1,1]*fisher_k[2,0])
        trace_fisher = fisher_k[0,0] + fisher_k[1,1] + fisher_k[2,2]
        print("\n fisher: \n", fisher_k)
        print("tr: ", trace_fisher)

        return u, x, fisher_k, trace_fisher, w1, w2, f1, f2

if __name__ == "__main__":
    fisher_prev = casadi.MX.zeros(3,3)
    x_init = 0.
    theta_init = np.pi, 
    xdot_init = 0., 
    thetadot_init = 0., 
    ctrl_init = 0.
    conv = 1e-6
    stateconcat = []
    ctrlconcat = []
    for t in range(100):
        try:
            opt = Traj_Optimization(l_pole_est = 0.5, 
                                m_cart_est = 0.4, 
                                m_pole_est = 0.7, 
                                th = 10, 
                                fisher_prev = fisher_prev, 
                                x_init= x_init,
                                theta_init = theta_init, 
                                xdot_init = xdot_init, 
                                thetadot_init = thetadot_init, 
                                ctrl_init = ctrl_init,
                                conv=conv)
            ctrl, state, fisher_prev, trace_fisher, w1, w2, f1, f2 = opt.run()
            current_iter = t
        except Exception as e:
            if 'Search_Direction_Becomes_Too_Small' in str(e):
                break
            elif 'Restoration_Failed' in str(e):
                break
            else:
                print(e)
                raise e
        fisher_prev = casadi.MX(fisher_prev)
        x_init = state[0,-1]
        theta_init = state[1,-1]
        xdot_init = state[2,-1]
        thetadot_init = state[3,-1]
        ctrl_init = ctrl[-1]
        conv = 1e-6
        print(t)
        print("weight_1: ", w1, "\n",
            "weight_2", w2, "\n",
            "freq_1: ", f1, "\n",
            "freq_2", f2, "\n")
        if t == 0:
            stateconcat = state
            ctrlconcat = ctrl
        else:
            stateconcat = np.concatenate((stateconcat, state), axis=1)
            ctrlconcat = np.concatenate((ctrlconcat, ctrl))
    np.savetxt('optctrl.txt', ctrlconcat, delimiter=',')
    np.savetxt('optstate.txt', stateconcat, delimiter=',')
    print("ctrl: \n",ctrlconcat, "\n state: \n", stateconcat)



