"""
All variables needed for system ID are in the following class
"""
try:
    import numpy as np
except Exception as e:
    raise e

class inits():

    def __init__(self) -> None:

        self.IC = {

            'x': 0.,                # initial horizontal position
            'theta':0.,             # initial angular position
            'xdot': np.pi,          # initial horizontal velocity
            'thetadot': 0.          # initial angular velocity

        }
        
        self.vars = {

            'l_pole': 0.5,          # Length of the pole
            'm_pole': 1.0,          # Mass of the pole
            'm_cart': 1.0,          # Mass of the cart
            'timestep': 0.001,      # timestep
            'gravity': 9.81,        # gravity
            'basepath': 'V2/',      # basepath to write/save text files
            'time_horizon': 1000    # time horizon for simulation

        }

        self.experiments = {

            'runs': 2,              # number of runs per trial
            'trials': 2             # number of trials

        }