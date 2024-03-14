try:   
    import numpy as np
    import numpy.random as nr
    import numpy.linalg as la
    import scipy as sp
    import casadi
    from tqdm import tqdm
    import time
    import os
    from cartpole_sim_V2 import cartpole as cp
    from param_est import param_est as pe
    from compute_fisher import fisherComputation as fc
    from traj_opt_V2 import Traj_Optimization
except Exception as e:
    raise e

class main():

    def __init__(self) -> None:
        self.trials =  5
        self.runs = 5
        self.restored_sim_time_horizon = 1000
        self.sim_time_horizon = 1000

    def random_main(self):

        """
        This runs the cartpole system with a set of controls with randomly uniform sample of amplitudes and frequencies.
        Parameter estimates are computed from control inputs and resulting trajectories from the simulator.
        The fisher information is computed and compounded over each run per trial
        """

        cp_instance = cp(None,None)
        pe_instance = pe(None,None,None,None)
        fc_instance = fc(None,None,None,None)
        inv_fisher_det_list = [[] for i in range(self.trials)]

        for trial in range(self.trials):

            print("trial ", trial+1)
            
            big_state = np.zeros((4,1))
            big_ctrl = np.zeros((1))
            fisher = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])

            for run in range(self.runs):

                print("run ", run+1)

                """
                Run the simulation
                """
                nr.seed(10*trial) # seeded differently every trial
                i = np.linspace(0.,self.sim_time_horizon, num=self.sim_time_horizon) # over the sim time horizon
                ctrl = (nr.uniform(-1,1,(1,1))*i*self.sim_time_horizon**-1)[0] # compute controls
                cp_instance = cp(self.sim_time_horizon, ctrl) # instantiate
                X, U = cp_instance.sim() # run
                cp_instance.store_to_txt_file(X,U) # store to a temporary text file

                """
                Append to "big" matrices and exec. sys. id.
                """
                state = pe_instance.get_state() # grab from the text file (not exactly needed as you can get this straight from X)
                ctrl = pe_instance.get_ctrl() # grab from the text file (not exactly needed as you can get this straight from U)
                
                # concatenation to previous states and controls
                if run == 0:
                    big_state = state
                    big_ctrl = ctrl
                else:
                    big_state = np.concatenate((big_state, state), axis=1)
                    big_ctrl =  np.concatenate((big_ctrl, ctrl))

                """
                Parameter Estimation
                """
                pe_instance = pe(big_state, big_ctrl, int(big_ctrl.shape[0]), self.sim_time_horizon) # instantiate
                l_pole_est, m_cart_est, m_pole_est = pe_instance.solve() # param est

                print("length of pole est: ", l_pole_est)
                print("mass of cart est: ", m_cart_est)
                print("mass of pole est: ", m_pole_est)

                """
                Compute the fisher information
                """
                fc_instance = fc(state, ctrl, state.shape[1], fisher) # instantiate
                fisher = fc_instance.run(l_pole_est, m_cart_est, m_pole_est) # compute
                det_fisher = la.det(fisher) # take the determinant
                inv_fisher_det_list[trial].append(1/det_fisher) # append the inverse fisher information (as it is invertible)

                print("det_fisher: ", det_fisher) # print resulting fisher information determinant

        return inv_fisher_det_list
    
    def opt_main(self):

        """
        This runs a cartpole system that estimates its parameters based on an optimal set of actions that maximizes its information about the parameters.
        Optimal controls are determined first, fed into the simulation, and the state control sets are used to estimate the parameters.
        Fisher information (inverse is a tight lower bound on posterior variance of params) is computed based on estimates and state control sets.
        """

        cp_instance = cp(None,None)
        pe_instance = pe(None,None,None,None)
        fc_instance = fc(None,None,None,None)
        inv_fisher_det_list = [[] for i in range(self.trials)]

        for trial in range(self.trials):

            print("trial ", trial+1)
            
            big_state = np.zeros((4,1))
            big_ctrl = np.zeros((1))
            fisher = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])

            l_pole_est = np.abs(nr.normal(loc=1.5, scale=1., size=(1,1)).item()) # sample from normal distribution, with arbitrarily high variance
            m_pole_est = np.abs(nr.normal(loc=1.5, scale=1., size=(1,1)).item()) # sample from normal distribution, with arbitrarily high variance
            m_cart_est = np.abs(nr.normal(loc=1.5, scale=1., size=(1,1)).item()) # sample from normal distribution, with arbitrarily high variance

            for run in range(self.runs):

                print("run ", run+1)

                """
                Trajectory Optimization
                """
                fisher_prev = casadi.MX.zeros(3,3)
                x_init = 0.
                theta_init = np.pi, 
                xdot_init = 0., 
                thetadot_init = 0., 
                ctrl_init = 0.
                conv = 1e-6
                th = 10
                print("Running Trajectory Optimization...")
                for t in tqdm(range(int(self.sim_time_horizon/th))):
                    try:
                        # instantiate
                        opt = Traj_Optimization(l_pole_est = l_pole_est, 
                                            m_cart_est = m_cart_est, 
                                            m_pole_est = m_pole_est, 
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
                        """
                        Exception handling:
                        Small search direction --> solver has reached an optimal solution and cannot take any large enough step forward
                        Restoration failed --> likely due to the linear solver used. Try using ma27 if the currant spral solver fails.
                        """
                        if 'Search_Direction_Becomes_Too_Small' in str(e):
                            self.sim_time_horizon = current_iter*th
                            break
                        elif 'Restoration_Failed' in str(e):
                            self.sim_time_horizon = current_iter*th
                            break
                        else:
                            print(e)
                            raise e
                        
                    """
                    Re-initialize the state, control, and other informative variables
                    """
                    fisher_prev = casadi.MX(fisher_prev)
                    x_init = state[0,-1]
                    theta_init = state[1,-1]
                    xdot_init = state[2,-1]
                    thetadot_init = state[3,-1]
                    ctrl_init = ctrl[-1]
                    conv = 1e-6

                    # concatenation to previous states and controls
                    if t == 0:
                        stateconcat = state
                        ctrlconcat = ctrl
                    else:
                        stateconcat = np.concatenate((stateconcat, state), axis=1)
                        ctrlconcat = np.concatenate((ctrlconcat, ctrl))

                """
                Run the simulation
                """
                print("Running Simulation...")
                cp_instance = cp(self.sim_time_horizon, ctrlconcat) # instantiate
                X, U = cp_instance.sim() # run the sim
                cp_instance.store_to_txt_file(X,U) # store to text files temporarily

                """
                Append to "big" matrices and exec. sys. id.
                """
                print("Getting Parameter Estimates")
                state = pe_instance.get_state() # grab from the text file (not exactly needed as you can get this straight from X)
                ctrl = pe_instance.get_ctrl() # grab from the text file (not exactly needed as you can get this straight from U)
                
                # concatenation to previous states and controls
                if run == 0:
                    big_state = state
                    big_ctrl = ctrl
                else:
                    big_state = np.concatenate((big_state, state), axis=1)
                    big_ctrl =  np.concatenate((big_ctrl, ctrl))

                pe_instance = pe(big_state, big_ctrl, int(big_ctrl.shape[0]), self.sim_time_horizon) # instantiate
                l_pole_est, m_cart_est, m_pole_est = pe_instance.solve() # param est

                print("length of pole est: ", l_pole_est)
                print("mass of cart est: ", m_cart_est)
                print("mass of pole est: ", m_pole_est)

                """
                Compute the fisher information
                """
                print("Fisher Computation...")
                fc_instance = fc(state, ctrl, state.shape[1], fisher) # instantiate
                fisher = fc_instance.run(l_pole_est, m_cart_est, m_pole_est) # compute
                det_fisher = la.det(fisher) # take determinant
                self.sim_time_horizon = self.restored_sim_time_horizon # reset the sim time horizon
                time.sleep(1)

                # print("fisher: ", fisher)
                print("det_fisher: ", det_fisher)
                l_pole_est = nr.normal(loc=l_pole_est,scale=1/det_fisher,size=(1,1)).item() # the new estimate is sampled from a normal distribution with the computed mean and approximate variance (or a tight lower bound on it rather)
                m_pole_est = nr.normal(loc=m_pole_est,scale=1/det_fisher,size=(1,1)).item() # the new estimate is sampled from a normal distribution with the computed mean and approximate variance (or a tight lower bound on it rather)
                m_cart_est = nr.normal(loc=m_cart_est,scale=1/det_fisher,size=(1,1)).item() # the new estimate is sampled from a normal distribution with the computed mean and approximate variance (or a tight lower bound on it rather)
                inv_fisher_det_list[trial].append(1/det_fisher) # append inverse fisher determinant
                
        return inv_fisher_det_list

m = main()

random_inv_fisher_list = m.random_main()
random_inv_fisher_list = np.array(random_inv_fisher_list).T
np.savetxt("random_inv_fisher_list.txt", random_inv_fisher_list, delimiter=',')

opt_inv_fisher_list = m.opt_main()
opt_inv_fisher_list = np.array(opt_inv_fisher_list).T
np.savetxt("opt_inv_fisher_list.txt", opt_inv_fisher_list, delimiter=',')



                