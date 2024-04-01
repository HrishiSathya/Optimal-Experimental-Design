try:   
    import numpy as np
    import numpy.random as nr
    import numpy.linalg as la
    import scipy as sp
    import casadi
    from tqdm import tqdm
    import time
    import os
    from inits import inits
    from utils import utils, noisy_sensor
    from sim import cartpole as cp
    from param_est import param_est as pe
    from fisher_computation import fisherComputation as fc
    from planning import Traj_Optimization
except Exception as e:
    raise e

class main(inits,utils,noisy_sensor):

    def __init__(self):
        super().__init__()

        self.trials =  self.experiments['runs']
        self.runs = self.experiments['trials']
        self.restored_sim_time_horizon = self.vars['time_horizon']
        self.sim_time_horizon = self.vars['time_horizon']
        self.utils_n = 3
        self.n_sampled_ctrls = self.utils_n
        self.n_normal_sampled_parameter_datapoints = None
        self.parameter_estimates = []
        self.seeded_values = None
        self.mean_u = 0.
        self.sigma_u = 20.
        self.mean_p = np.array([5., 5., 5.])
        self.sigma_p = np.array([5., 5., 5.])
        self.state_list = []
        self.ctrl_list = []
        self.p = 3
        self.m = self.n_sampled_ctrls
        self.noise = 0.01

    def random_main(self):

        """
        method for experiments with random sampled controls
        """

        # initialize the fisher information determinant list
        inv_fisher_det_list = [[] for _ in range(self.trials)]
        cov_det_list = [[] for _ in range(self.trials)]
        for trial in range(self.trials):
            
            """
            iterate through every trial
            """
            print('trial: ', trial + 1)

            # set values back to zero
            States, Ctrl, Fisher_Inf = self.reset()
            self.sim_time_horizon = self.restored_sim_time_horizon
            fisher_list = []

            for run in range(self.runs):
                
                """
                iterate through every run
                """
                print('run: ', run + 1)

                # re-initialize the state_list and control_list
                self.state_list = []
                self.ctrl_list = []

                """
                iterate through every set of control samples.
                Sample from p(u) = N(mu_u,sigma_u)
                """
                # choose the seed
                self.seeded_values = run + trial**2 + 10
                seed = int(self.seeded_values)

                # get samples of controls
                sampled_ctrls = self.normal_sample(self.sim_time_horizon,
                                                    self.n_sampled_ctrls,
                                                    seed,
                                                    self.mean_u,
                                                    self.sigma_u)
                
                for n in range(self.n_sampled_ctrls):

                    """
                    Run the simulation.
                    p(X | u) p(u)
                    """
                    cartpole_sim = cp(sampled_ctrls[n,:])
                    X_measured = self.get_noisy_trajectories(cartpole_sim.sim(), seed)

                    """
                    reshape and append controls and shape
                    """
                    if run == 0:
                        States[n] = X_measured
                        Ctrl[n] = sampled_ctrls[n,:]
                    else:
                        States[n] = np.concatenate((States[n], X_measured), axis=1)
                        Ctrl[n] = np.concatenate((Ctrl[n], sampled_ctrls[n,:]))
                    self.state_list.append(States[n])
                    self.ctrl_list.append(Ctrl[n])

                    """
                    Estimate the parameters from states and controls.
                    p(Theta | X_measured, u_sampled)
                    Theta* = max_Theta log(Pi(p(Yj | Xj, uj, Theta)))
                    """
                    param_estimator = pe(States[n], Ctrl[n], self.noise, States[n].shape[1])
                    mass_pole, length_pole, mass_cart = param_estimator.solve()
                    self.parameter_estimates.append([mass_pole, length_pole, mass_cart])
                    print('estimate ', int(n+1), 
                          ': m_p = ', mass_pole, 
                          ', l_p = ', length_pole, 
                          ', m_c = ', mass_cart)
                    
                    """
                    Fisher information
                    """
                    fisher_computation = fc(X_measured, sampled_ctrls[n,:], mass_pole, length_pole, mass_cart, Fisher_Inf, self.noise)
                    ###
                    fisher_list.append(fisher_computation.run())
                    
                """
                Covariance Matrix from states, controls, and parameters
                Calculated empirically
                """
                Cov_mat = self.get_covariance_matrix(np.array(self.parameter_estimates))
                print("Covariance Matrix: \n ", Cov_mat)
                print("\n det: \n", np.linalg.det(Cov_mat))

                """
                Sum all fisher inf.
                """
                Fisher_Inf = np.array(sum(fisher_list))
                inv_fisher_sum = np.linalg.inv(Fisher_Inf)
                print("\n Inverse Fisher Information: \n ", inv_fisher_sum)
                print("\n det: \n", np.linalg.det(inv_fisher_sum))


m = main()
rand = m.random_main()




