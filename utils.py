try:
    import numpy as np
    import casadi
    from inits import inits
except Exception as e:
    raise e

class utils():

    """
    the shape of the dataset is #timeiters(row) x #params(cols)

    dataset = [[x11, x21, ..., xp1], 
                [x12, x22, ..., xp2], 
                        ...
                [x1m, x2m, ..., xpm]]
    """

    def __init__(self):
        
        self.p = None         
        self.m = None
        self.utils_n = 10       

    def __get_average(self, dataset):
        
        """
        returns the average of each datapoint for each parameter
        """
        D_ave = np.zeros((self.p,1))
        for param in range(self.p):
            D_ave[param] = np.average(dataset[:,param])

        return D_ave

    def __get_variance(self, dataset):
        
        """
        returns the variance of each parameter in the dataset
        """
        # initialize parameters
        sumsquare = 0
        variance = np.zeros((self.p,1))

        # get average of each parameter from dataset
        D_ave = self.__get_average(dataset)
        
        # sum over all datapoints for every parameter, and obtain variance
        for param in range(self.p):
            for sample in range(self.m):
                sumsquare += (dataset[sample,param] - D_ave[param])**2
            variance[param] = sumsquare
        variance *= (self.m)**-1

        return variance
    
    def __get_covariance(self, parameter1_data, parameter2_data):

        """
        returns the covariance matrix
        """

        # initialize, reshape, get average from parameter-specific data
        sumsquare = 0
        parameter1_data = parameter1_data.reshape(int(self.m),1)
        parameter2_data = parameter2_data.reshape(int(self.m),1)
        parameter1_data_ave = np.average(parameter1_data)
        parameter2_data_ave = np.average(parameter2_data)

        # obtain covariance
        for sample in range(int(self.m)):
            sumsquare += (parameter1_data[sample] - parameter1_data_ave) * (parameter2_data[sample] - parameter2_data_ave)
        covariance = sumsquare/(float(self.m))

        return covariance

    
    def get_covariance_matrix(self, dataset):

        """
        returns covariance matrix
        """

        # check if the dataset is a numpy type
        if not isinstance(dataset, (np.ndarray, np.generic)):
            raise Exception('dataset is not numpy datatype')

        # initialize
        self.m = float(dataset.shape[0])
        self.p = float(dataset.shape[1])
        Cov_matrix = np.zeros((int(self.p), int(self.p)))

        # obtain covariance matrix
        for param_row in range(Cov_matrix.shape[0]):
            for param_col in range(Cov_matrix.shape[1]):
                if param_row == param_col:
                    Cov_matrix[param_row,param_col] = self.__get_covariance(dataset[:,param_row], dataset[:,param_col])
        
        return Cov_matrix
    
    def get_determinant(self, matrix):
        
        """
        returns the determinant of a matrix
        """

        # check type and size of the matrix
        if not isinstance(matrix, (np.ndarray, np.generic)):
            raise Exception('dataset is not numpy datatype')
        elif matrix.shape[0] != matrix.shape[1]:
            raise Exception('not square matrix')
        
        return np.linalg.det(matrix)

    
    def get_average(self, dataset, axis):

        """
        returns the average value of a dataset along a certain axis
        """

        # check axis value and type for dataset
        if axis != 0 and axis != 1: 
            raise Exception('axis not 0 or 1')
        if not isinstance(dataset,(np.ndarray,np.generic)):
            raise Exception('dataset not numpy type')

        # obtain average
        ave = []
        split_data = np.split(dataset, dataset.shape[axis], axis=axis)

        for _ in split_data:
            ave.append(np.average(split_data))

        return np.array(ave)
    
    def get_average_state(self, *args):

        """
        returns averaged trajectory of all trajectories
        """
        # check type of states
        for state in args[0]:
            if not isinstance(state,(np.ndarray,np.generic)):
                raise Exception('state not a numpy type')
            elif state.shape[0] != 4:
                print('state.shape = ', state.shape)
                raise Exception('state not of shape (4,N)')
                
        # concatenate all state variables
        counter = 0
        for state in args[0]:
            if counter == 0:
                x_stack = state[0,:]
                theta_stack = state[1,:]
                xdot_stack = state[2,:]
                thetadot_stack = state[3,:]
            else:
                x_stack = np.vstack((x_stack,state[0,:]))
                theta_stack = np.vstack((theta_stack,state[1,:]))
                xdot_stack = np.vstack((xdot_stack,state[2,:]))
                thetadot_stack = np.vstack((thetadot_stack,state[3,:]))
            counter += 1
        
        # obtain average
        x_ave = self.get_average(x_stack,axis=1)
        theta_ave = self.get_average(theta_stack,axis=1)
        xdot_ave = self.get_average(xdot_stack,axis=1)
        thetadot_ave = self.get_average(thetadot_stack,axis=1)

        return np.array([x_ave, theta_ave, xdot_ave, thetadot_ave]).reshape(4,args[0][0].shape[1])

    def get_average_control(self, *args):

        """
        returns averaged control inputs of all control inputs
        """
        # check type of ctrl inputs
        for ctrl in args[0]:
            if not isinstance(ctrl,(np.ndarray,np.generic)):
                raise Exception('ctrl not a numpy type')
        
        # concatenate all controls
        counter = 0
        for ctrl in args[0]:
            if counter == 0:
                ctrl_stack = ctrl
            else:
                ctrl_stack = np.vstack((ctrl_stack,ctrl))
            counter += 1
        
        # obtain average
        ctrl_ave = self.get_average(ctrl_stack,axis=1)

        return ctrl_ave
    
    def normal_sample(self,n_samples,n_trajs,seed,mean,std):
        
        """
        returns n random samples from a normal distribution 
        """
        # seed a value
        np.random.seed(seed)

        return np.random.normal(mean,std,size=(n_trajs,n_samples))
    
    def reset(self):

        """
        this resets state, controls, and fisher information to zero vector/matrix
        """
        state = [[] for _ in range(self.utils_n)]
        ctrl = [[] for _ in range(self.utils_n)]
        fisher_inf = np.zeros((int(self.p),int(self.p)))

        return state, ctrl, fisher_inf
    
class noisy_sensor():

    def __init__(self):

        self.noise = None

    def __get_normal_samples(self,n_samples,seed,mean,std):
        
        """
        returns n random samples from a normal distribution 
        """
        # seed a value
        np.random.seed(seed)

        return np.random.normal(mean,std,size=(n_samples,1))
    
    def get_noisy_trajectories(self, Xbar, seed):
        
        """
        returns noisy trajectory values
        """
        # check type of Xbar
        if not isinstance(Xbar, (np.ndarray, np.generic)):
            raise Exception('Xbar not numpy type')
        elif Xbar.shape[0] != 4:
            print('Xbar.shape = ', Xbar.shape)
            raise Exception('Xbar not of shape (4,N)')
        
        # add noise to your trajectories
        X = np.zeros(Xbar.shape)
        for t in range(Xbar.shape[1]):
            X[0,t] = self.__get_normal_samples(1, seed, Xbar[0,t], self.noise).item()
            X[1,t] = self.__get_normal_samples(1, seed, Xbar[1,t], self.noise).item()
            X[2,t] = self.__get_normal_samples(1, seed, Xbar[2,t], self.noise).item()
            X[3,t] = self.__get_normal_samples(1, seed, Xbar[3,t], self.noise).item()
        
        return X
    
# x = utils()
# p = np.array([[.18, .50, 0.99]])
# print(x.get_covariance_matrix(p))